# SPDX-License-Identifier: Apache-2.0
"""PIVOT-Refine indexer (simple, reliable core).

For a grouped MTP decode batch (g queries per request, g = spec tokens + 1)
this replaces the per-query full-prefix indexer scan with one shared
mean-proxy coarse screen plus a per-query top-k refine over the candidates:

  1. mean proxy  q_bar = mean(q[group])          (torch, graph-safe)
  2. coarse      C = top-4096 proxy scores over each prefix, computed in
     screen      torch (not the native op, so the superset can be 4096 --
                 the paper's number -- instead of the native 2048 limit)
  3. refine      broadcast C to each query row, score with the native
                 formula, top-k (k = 2048)

Mixed-batch routing (PD co-location): every batch is reordered to
decode -> short_extend -> long_extend -> prefill
(reorder_batch_to_split_decodes_and_prefills, driven by this backend's
reorder_batch_threshold = 1 + num_speculative_tokens), so the decode
requests form a contiguous HEAD of the batch. select_topk routes that head
through PIVOT and runs the prefill tail through the native indexer, then
concatenates -- decode requests get PIVOT in every step they appear in, not
only in all-decode steps.

Lossless contract: with L + g <= 2048 the refine output reproduces the whole
prefix exactly; with L + g <= 4096 the coarse screen still returns the whole
prefix, so refine only drops keys past 2048 by score. k is fixed at 2048 (the
native sparse_count consumed by npu_sparse_flash_attention); there is
deliberately no k knob.

BF16 only: enable_sparse_li_c8 falls back to the native indexer.

Grouping is definitional: actual_seq_lengths_query is the cumulative
per-request query-token count (vLLM v1 query_start_loc[1:]) that the native
indexer and the sparse-attention kernel both use to group TND rows into
requests. The segment split needs a few scalar device->CPU syncs (int()/bool()),
acceptable in the eager decode path -- no graph capture.
"""

from __future__ import annotations

import dataclasses

import torch

from vllm.logger import logger
from vllm_ascend import envs

# PIVOT-Refine budget split. The proxy screen returns a _COARSE_BUDGET
# candidate superset (4096, the paper's number); per-query refine narrows to
# _REFINE_BUDGET top-k (2048, the native sparse_count consumed by
# npu_sparse_flash_attention). The superset is 2x the refine width so the
# mean-proxy approximation still covers each query's true top-k.
_COARSE_BUDGET = 4096
_REFINE_BUDGET = 2048

# Precision diagnostics compiled out: the lossless self-check walks every
# output row with per-element int() device->CPU syncs (N x 2048 per call,
# per layer, per step), which dominates decode step time. Flip to True on
# the NPU box when a precision problem needs localizing; the three
# PIVOT[dbg] log lines come back.
_ENABLE_REPORT = False

# A decode group carries 1 + num_speculative_tokens query rows, and the SFA
# decode threshold is asserted <= 16 (TND layout limit), so a leading run
# with per-request counts in [2, 16] can only be decode groups or
# <= 16-token prefills. The latter are exactly reproducible under PIVOT
# (their whole prefix fits the lossless window), so admitting them is safe.
_MAX_GROUP = 16

# Real-data capture state (VLLM_ASCEND_PIVOT_REFINE_DUMP): bounded count, no
# queue, no background thread -- each capture is one synchronous torch.save
# of a few MB. STRIDE samples every Nth hit so the MAX quota spreads over the
# whole run (first-N-hits-only would miss the long-prefix windows entirely).
# Diagnostics only.
_dump_count = 0
_dump_hits = 0
_dump_layer = None  # pinned at first dump-eligible invocation (or from env)
# Distinct indexer-layer names first seen (only indexer layers ever reach the
# dump code, so this doubles as the model's real indexer-layer roster when an
# explicit DUMP_LAYER guess is wrong).
_dump_seen: list[str] = []
_dump_miss_warned = False


def _local_rank() -> str:
    """TP rank for dump sharding (multi-rank writes must not collide)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:
        pass
    import os

    return os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))


def _layer_num(name: str) -> int | None:
    """Last integer run in a layer name, or None if it has no digits.

    Model layer prefixes carry an index (e.g. ``model.layers.7`` or
    ``model.layers.7.self_attn``); indexer layers execute in ascending index
    order each decode step. Used to detect when an explicit DUMP_LAYER guess
    names a layer that does not exist (we pass its index position without ever
    seeing it) so the dump can warn and fall back instead of capturing nothing.
    """
    i = len(name) - 1
    while i >= 0 and not name[i].isdigit():
        i -= 1
    if i < 0:
        return None
    j = i
    while j >= 0 and name[j].isdigit():
        j -= 1
    return int(name[j + 1:i + 1])


def _dump_gate(layer_name: str) -> bool:
    """Decide whether to capture THIS (step, layer) invocation.

    Combines the one-layer filter and the stride sampler. Only indexer layers
    ever call in (sfa_v1.py:1496 has_indexer hard gate), so layer_name is
    always a real indexer layer; an explicit DUMP_LAYER that names no indexer
    layer logs the real roster and falls back to the current layer instead of
    silently capturing nothing. Diagnostics only -- never raises.
    """
    global _dump_hits, _dump_layer, _dump_seen, _dump_miss_warned
    if not envs.VLLM_ASCEND_PIVOT_REFINE_DUMP:
        return False
    if _dump_count >= envs.VLLM_ASCEND_PIVOT_REFINE_DUMP_MAX:
        return False
    if _dump_layer is None:
        want = envs.VLLM_ASCEND_PIVOT_REFINE_DUMP_LAYER
        if want == "first" or want in layer_name:
            _dump_layer = layer_name
        elif want != "first":
            # _dump_seen doubles as the model's real indexer-layer roster.
            if layer_name not in _dump_seen:
                _dump_seen.append(layer_name)
                logger.info_once(
                    "PIVOT dump: DUMP_LAYER=%r not matched yet; indexer "
                    "layer %s present (real indexer layers so far: %s)",
                    want, layer_name, ", ".join(_dump_seen))
            # Layers run in ascending index order each step: once a layer
            # numerically past the target has appeared, the target (a dense
            # layer) can never come. Fall back to the current real indexer
            # layer so the run still captures data.
            tw = _layer_num(want)
            if (not _dump_miss_warned and tw is not None
                    and any(_layer_num(l) is not None and _layer_num(l) > tw
                            for l in _dump_seen)):
                _dump_miss_warned = True
                logger.warning(
                    "PIVOT dump: DUMP_LAYER=%r matches no indexer layer "
                    "(real ones seen: %s); falling back to %s so the run "
                    "still captures data.",
                    want, ", ".join(_dump_seen), layer_name)
                _dump_layer = layer_name
    if _dump_layer != layer_name:
        return False
    _dump_hits += 1
    return (_dump_hits - 1) % envs.VLLM_ASCEND_PIVOT_REFINE_DUMP_STRIDE == 0


def _dump_real_inputs(q_dq, weights, C, req_ids, aslq, aslk, kv_cache,
                      block_table, block_size, layer_hint, op_topk,
                      aslk_op=None):
    """Persist one real op invocation + BOTH outputs for offline replay.

    Saves the exact tensors the op consumes (query/weights/candidates/aslq/
    aslk) plus the KV rows the op can reach (only slots referenced by
    candidates, ~c rows per request) and the REAL block table, so the replay
    harness rebuilds the production PA layout faithfully. Saves BOTH
    outputs for the same inputs:
      - op_topk: the op's raw answer ([D, 1, 2048] candidate COLUMN indices,
        before the caller gather) -- replaying the same inputs and diffing
        against this catches nondeterminism/races bit-exactly;
      - ref_topk: the python _refine_topk answer (positions) -- the
        precision reference the op is supposed to match after gather.
    TP safe: files land in <dir>/rank<N>/ so concurrent ranks never
    overwrite each other (each rank's inputs differ under TP).
    Never raises -- diagnostics must not take down the decode path.

    aslk vs aslk_op: `aslk` is the RAW seq length (L+g) kept for the replay
    oob bound / reporting; `aslk_op` is the CLAMPED value the op actually
    received (clamp to C row width -- the op's S2 loop walks the candidate
    list over [0, aslk), so an unclamped L+g > C.shape[1] reads past the row
    end). The replay MUST feed aslk_op back, not the raw aslk; persisting the
    exact consumed value here is what makes the determinism diff meaningful.
    """
    global _dump_count
    import os

    _dump_count += 1
    idx = _dump_count
    try:
        out_dir = os.path.join(
            envs.VLLM_ASCEND_PIVOT_REFINE_DUMP_DIR, f"rank{_local_rank()}")
        os.makedirs(out_dir, exist_ok=True)
        # Token-major rows of the PA_BSND K cache, flattened EXACTLY as
        # _refine_topk consumes it (view(-1, Dh), slot = block_table[r,
        # pos//bs]*bs + pos%bs). NOT view(kv_cache[2].shape[0], -1): that
        # flattens on the BLOCK axis and leaves each row a whole block
        # (bs * Dh wide), which the replay rebuild (token rows of Dh) cannot
        # scatter back -- that shape mismatch was the dump bug.
        kc = kv_cache[2].view(-1, kv_cache[2].shape[-1])  # [B*bs, Dh]
        # slots referenced by valid candidate positions (per request)
        c64 = C.to(torch.int64).clamp(min=0)
        blk = (c64 // block_size)
        slots = (block_table.to(torch.int64).gather(1, blk) * block_size
                 + c64 % block_size).reshape(-1)
        slots = slots.unique()
        ref = _refine_topk(q_dq, weights, C, req_ids, kv_cache, block_table,
                           block_size, q_dq.shape[0])
        if aslk_op is None:
            aslk_op = torch.clamp(aslk, max=C.shape[1])
        torch.save(
            {
                "q_dq": q_dq.cpu(), "weights": weights.cpu(),
                "C": C.cpu(), "aslq": aslq.cpu(), "aslk": aslk.cpu(),
                # op 实收的 clamp 后 aslk(见 docstring)——replay 必须喂这个,
                # 不是上面 raw 的 aslk(L+g > C 宽时两者分叉)。
                "aslk_op": aslk_op.cpu(),
                "block_table": block_table.cpu(), "block_size": block_size,
                "kv_slots": slots.cpu(), "kv_rows": kc[slots].cpu(),
                "kv_num_rows": kc.shape[0],
                "op_topk": op_topk.detach().cpu(),  # raw cols, pre-gather
                "ref_topk": ref.cpu(), "layer": layer_hint,
            },
            os.path.join(out_dir, f"cap_{idx:04d}.pt"))
        logger.info("PIVOT refine dump %d saved to %s (layer=%s, D=%d)",
                    idx, out_dir, layer_hint, q_dq.shape[0])
    except Exception as e:
        logger.warning("PIVOT refine dump %d failed: %s", idx, e)


def _capturing() -> bool:
    try:
        return torch.npu.is_current_stream_capturing()
    except Exception:
        return False


def _request_counts(cum: torch.Tensor) -> torch.Tensor:
    """Per-request query-token counts from cumulative ends (query_start_loc[1:])."""
    starts = torch.cat([torch.zeros(1, dtype=cum.dtype, device=cum.device), cum[:-1]])
    return (cum - starts).to(torch.int64)


class PivotIndexer:
    """PIVOT-Refine top-k selection (op-reuse core, BF16)."""

    @staticmethod
    def select_topk(
        sfa_impl,
        q_li: torch.Tensor,
        q_li_scale: torch.Tensor | None,
        q_li_shape_ori: tuple | None,
        weights: torch.Tensor,
        kv_cache: tuple,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        allow_whole_batch: bool = True,
    ) -> torch.Tensor | None:
        """Return topk_indices [N_in, 1, 2048] (0-based logical key positions).

        Decode requests sit at the head of the batch (the engine reorders to
        decode -> ... -> prefill), so the leading run of requests with a
        uniform query count g is the decode segment: requests [0, K), query
        rows [0, D). That segment runs through PIVOT; a non-empty prefill
        tail runs through the native indexer and the results are
        concatenated.

        Returns None when the batch has no grouped decode head (C8, g < 2,
        g > 16, or a uniform batch in a prefill state when the caller
        disallows whole-batch PIVOT); the caller then falls back to the
        native indexer for the whole batch.
        """
        if getattr(sfa_impl, "enable_sparse_li_c8", False):
            logger.warning_once(
                "PIVOT: enable_sparse_li_c8 is set; falling back to the "
                "native indexer (PIVOT supports the BF16 path only)."
            )
            return None

        cum = actual_seq_lengths_query  # [R], cumulative ends
        seq_lens = actual_seq_lengths_key  # [R], == L + g at indexer time
        R_all = cum.shape[0]
        if R_all == 0:
            return None

        N = attn_metadata.num_actual_tokens
        if int(cum[-1]) != N:
            # TND row count must match the last cumulative boundary; if not,
            # the caller's grouping is not what we assume -- bail to native.
            logger.warning_once(
                "PIVOT: actual_seq_lengths_query[-1]=%d != num_actual_tokens=%d; "
                "bailing the whole batch to the native indexer.", int(cum[-1]), N)
            return None

        counts = _request_counts(cum)  # [R]
        g = int(counts[0])
        if not 2 <= g <= _MAX_GROUP:
            # Not a grouped-decode head (single-token decode, or a prefill
            # leading the batch): nothing to amortize.
            logger.warning_once(
                "PIVOT: leading request carries g=%d query rows (not a "
                "grouped-decode head); bailing the whole batch to the native "
                "indexer. If this fires in PD co-location, the decode->prefill "
                "reorder did not put decodes at the batch head.", g)
            return None

        # Decode segment: the leading run of requests carrying g query rows.
        eq = counts == g
        if bool(eq.all()):
            K, D = R_all, N
        else:
            K = int((~eq).nonzero()[0, 0])  # first non-group request
            D = int(cum[K - 1])

        if K == R_all and not allow_whole_batch:
            # A uniform batch in a prefill state is an all-prefill shape
            # (needs prefix caching / chunked prefill to arise): PIVOT's
            # proxy adds nothing there and could approximate past the
            # lossless window, so keep it on the native indexer.
            logger.warning_once(
                "PIVOT: uniform batch (g=%d) in attn_state=%s; keeping the "
                "whole batch on the native indexer.", g,
                attn_metadata.attn_state)
            return None

        device = q_li.device
        N_in = q_li.shape[0]
        q_dq = q_li[:D]  # raw BF16 [D, H, Dh] (no hadamard/quant on this path)

        # ---- 1. mean proxy (segment mean over each request's g queries) --
        H, Dh = q_dq.shape[1], q_dq.shape[2]
        q_bar = q_dq.view(K, g, H, Dh).mean(dim=1)  # [K, H, Dh]
        w_bar = weights[:D].view(K, g, H).mean(dim=1)  # [K, H]

        # ---- 2. coarse screen: torch proxy scan, K rows -> 4096 ----------
        # Done in torch (not the native npu_lightning_indexer) so the
        # candidate superset is _COARSE_BUDGET (4096, the paper's number)
        # rather than the native 2048 sparse_count hard limit. Same score
        # formula (sum_h w_bar * relu(q_bar . k)); the SFA attention kernel
        # applies causality downstream. The proxy domain is [0, L) -- the
        # paper (Eq. 5) recalls the pool at the group's FIRST position, so
        # this step's own g tokens never enter the proxy pool (they are the
        # worst-represented entries of a mean proxy anyway); the local
        # window below supplies them to the refine domain instead.
        # Lossless fast path: if every decode request's prefix length
        # L = seq_lens - g is within _COARSE_BUDGET, the coarse top-4096 over
        # [0, L) already covers the whole prefix (top-4096 over < 4096
        # distinct positions is the whole set), so the proxy bmm + topk are
        # pure overhead -- build the full-prefix candidate set directly.
        # Ascending (not score-descending) column order is irrelevant
        # downstream: the refine op re-sorts by score, so the emitted
        # topk_indices are bit-identical to the coarse-screened path
        # (candidate set [0, L+g) unchanged).
        L = seq_lens[:K].to(torch.int64) - g  # [K] per-request prefix length
        if int(L.max()) <= _COARSE_BUDGET:
            col = torch.arange(
                _COARSE_BUDGET, dtype=torch.int64, device=device)
            C = torch.where(col[None, :] < L[:, None], col[None, :], -1)
        else:
            C = _coarse_screen(
                q_bar, w_bar, kv_cache, attn_metadata.block_table[:K],
                attn_metadata.block_size, seq_lens[:K] - g,
            )

        # ---- 2b. per-query local window (paper Appendix B, decode) -------
        # The paper's decode refine domain is (pool U W_t) per query, with
        # W_t = [t-W+1, t] and W >= g so the window covers every token
        # generated within the step. The op's candidate row is shared per
        # request, so the window enters the row as the GROUP's window union
        # [L-g+1, L+g) -- deduped against C, appended, C widened (the op
        # imposes no width bound; workspace scales linearly), then the valid
        # candidates are COMPACTED to the row front (see _inject_local_window)
        # so the op's S2 walk [0, aslk) reaches the whole refine domain.
        # Window entries then COMPETE BY SCORE exactly like pool entries --
        # the paper's decode semantics, not a forced-in reserve slot. Each
        # row's causal mask (SFA kernel) trims the union to that row's own
        # [t-g+1, t]. W = g reproduces the paper's experimental
        # configuration (w = 4 = g).
        if envs.VLLM_ASCEND_PIVOT_LOCAL_WINDOW:
            C, aslk_op = _inject_local_window(C, seq_lens[:K], g)
        else:
            # aslk drives the per-request S2 chunk loop over the CANDIDATE
            # LIST (row width = C.shape[1]). In the truncated region
            # L+g > 4096 an unclamped aslk makes the kernel read candidate
            # columns past the row end (garbage/adjacent rows); clamp.
            aslk_op = torch.clamp(seq_lens[:K], max=C.shape[1])

        # ---- 3. refine: broadcast C, score, top-k -------------------------
        # Query row -> request id within the decode segment (the leading run
        # is uniform, so this equals repeat_interleave(arange(K), g)).
        req_ids = torch.repeat_interleave(
            torch.arange(K, dtype=torch.int64, device=device), counts[:K]
        )  # [D]
        if envs.VLLM_ASCEND_PIVOT_REFINE_USE_OP:
            # Validated op (NPU 9/9 tie-aware PASS). The op groups the
            # per-request candidates C via the cumulative aslq internally.
            # Fall back to the torch reference on any op failure so the
            # decode path keeps serving.
            try:
                # Op prototype requires candidates DT_INT32; _coarse_screen
                # returns torch.topk indices (int64). Cast only on the op
                # path -- the torch reference below consumes C as-is.
                # aslk_op (computed above = the per-row valid candidate count
                # after local-window compaction; = L+g in the lossless region,
                # up to 4096+2g-1 once truncated) bounds the per-request S2
                # chunk loop over the CANDIDATE LIST; the DUMP persists the
                # same value the op actually consumed.
                topk_dec = torch.ops._C_ascend.npu_indexer_refine(
                    q_dq,
                    kv_cache[2],
                    weights[:D],
                    C.to(torch.int32),
                    actual_seq_lengths_query=cum[:K],
                    actual_seq_lengths_key=aslk_op,
                    block_table=attn_metadata.block_table[:K],
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=_REFINE_BUDGET,
                )  # [D, 1, _REFINE_BUDGET] int32
                # One-shot confirmation that the op path (not the torch
                # fallback below) is live -- without this a silently failing
                # op would make gsm8k runs indistinguishable from USE_OP=0.
                logger.info_once(
                    "PIVOT refine: using npu_indexer_refine op "
                    "(VLLM_ASCEND_PIVOT_REFINE_USE_OP=1).")
                # Dump is diagnostics: guarded so a dump bug can never read
                # "op failed", never trigger the torch fallback, never break
                # the decode path. _dump_real_inputs self-protects too.
                _do_dump = False
                _layer = getattr(sfa_impl, "layer_name", "unknown")
                if envs.VLLM_ASCEND_PIVOT_REFINE_DUMP:
                    try:
                        _do_dump = _dump_gate(_layer)
                    except Exception as e:
                        logger.warning_once(
                            "PIVOT refine dump gate error (%s); skipping dump", e)
                if _do_dump:
                    # Dump AFTER the op returned, so the capture holds inputs
                    # + the op's raw output (pre-gather cols) + the python
                    # reference output computed from the same tensors.
                    _dump_real_inputs(
                        q_dq, weights[:D], C, req_ids, cum[:K], seq_lens[:K],
                        kv_cache, attn_metadata.block_table[:K],
                        attn_metadata.block_size,
                        _layer,
                        op_topk=topk_dec,
                        # op 实收的 aslk(=该行有效候选数,经 local-window
                        # 加宽+compact 后与 raw seq_lens[:K] 分叉),回放必须
                        # 用这个。
                        aslk_op=aslk_op,
                    )
                # CRITICAL: the refine op's S2 axis is the CANDIDATE LIST, so
                # it sorts and emits the candidate COLUMN index -- unlike the
                # native indexer where column == KV position. _coarse_screen
                # returns score-ordered positions (C[r, j] = j-th best
                # position, NOT identity), so a raw column value is useless to
                # the SFA kernel unless mapped back to the position value.
                # Missing this gather is why the op path read wrong keys and
                # produced the massive repetition loops in gsm8k.
                cols = topk_dec.view(D, _REFINE_BUDGET).to(torch.int64)
                pos = C[req_ids].gather(1, cols.clamp(min=0))  # [D, 2048]
                topk_dec = (
                    torch.where(cols < 0, -1, pos)
                    .view(D, 1, _REFINE_BUDGET)
                    .to(torch.int32)
                )
            except Exception as e:
                logger.warning("PIVOT refine op/gather failed, recomputing via torch: %s", e)
                topk_dec = _refine_topk(
                    q_dq,
                    weights[:D],
                    C,
                    req_ids,
                    kv_cache,
                    attn_metadata.block_table[:K],
                    attn_metadata.block_size,
                    D,
                )
        else:
            topk_dec = _refine_topk(
                q_dq,
                weights[:D],
                C,
                req_ids,
                kv_cache,
                attn_metadata.block_table[:K],
                attn_metadata.block_size,
                D,
            )

        # ---- 4. causality is enforced by the SFA kernel, not here ----------
        # sparse_mode=3 gives the SFA kernel a per-row threshold
        # (nextTokensPerBatch + gS1Idx/gSize + 1 == L+i+1, the query row's
        # causal prefix) that it enforces in all three consumption points
        # (CalcSinnerTopKBegin / CalcTopKBlockInfo / CopyInKv): positions >=
        # L+i+1 are skipped, never attended. Do NOT inject -1 sentinels here
        # -- the kernel breaks its sparse scan at the first -1
        # (kernel_mla.h CalcSinnerTopKBegin), so a mid-list -1 silently drops
        # every later valid candidate and collapses attention (this was the
        # repetition seen even on the torch path). Both paths emit valid
        # positions in score order; the kernel handles causality.

        if K == R_all:
            topk_indices = topk_dec
        else:
            # Mixed batch: the prefill tail (requests [K, R_all), rows
            # [D, N)) stays on the native indexer.
            topk_indices = torch.cat(
                [
                    topk_dec,
                    _native_indexer_tail(
                        sfa_impl,
                        q_li,
                        q_li_scale,
                        q_li_shape_ori,
                        weights,
                        kv_cache,
                        attn_metadata,
                        actual_seq_lengths_query,
                        actual_seq_lengths_key,
                        K,
                        D,
                        N,
                    ),
                ],
                dim=0,
            )

        # Graph padding guard: padded rows [N, N_in) get -1 tails so the
        # output row count matches the native path (num_input_tokens).
        if N_in > N:
            row_pad = torch.full(
                (N_in - N, 1, topk_indices.shape[-1]),
                -1,
                dtype=topk_indices.dtype,
                device=device,
            )
            topk_indices = torch.cat([topk_indices, row_pad], dim=0)

        # use_index_cache width guard: the read side returns the full buffer
        # width, so pad the output to the buffer width with -1 tails.
        if getattr(sfa_impl, "use_index_cache", False) and sfa_impl.topk_indices_buffer is not None:
            buf_width = sfa_impl.topk_indices_buffer.shape[-1]
            if topk_indices.shape[-1] < buf_width:
                pad = torch.full(
                    (topk_indices.shape[0], 1, buf_width - topk_indices.shape[-1]),
                    -1,
                    dtype=topk_indices.dtype,
                    device=device,
                )
                topk_indices = torch.cat([topk_indices, pad], dim=-1)

        if _ENABLE_REPORT and not _capturing():
            try:
                _report(seq_lens[:K], counts[:K], C, topk_indices, D, K, g,
                        R_all, N, aslk=aslk_op)
            except Exception as e:  # diagnostics must never take down the decode path
                logger.warning("PIVOT[dbg] _report failed: %s", e)

        logger.debug(
            "PIVOT refine: rows=%d/%d reqs=%d/%d g=%d", D, N, K, R_all, g
        )
        return topk_indices


def _native_indexer_tail(
    sfa_impl,
    q_li: torch.Tensor,
    q_li_scale: torch.Tensor | None,
    q_li_shape_ori: tuple | None,
    weights: torch.Tensor,
    kv_cache: tuple,
    attn_metadata,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    K: int,
    D: int,
    N: int,
) -> torch.Tensor:
    """Native indexer over the prefill tail: requests [K, R), rows [D, N).

    The tail is a self-contained sub-batch: query rows [D, ...), cumulative
    query lens rebased by -D, and the block table / key lens sliced to the
    tail requests -- exactly what the native indexer would have received
    for those requests alone.
    """
    # Imported lazily: vllm_ascend.device.device_op pulls torch_npu, which is
    # absent on non-NPU hosts (CPU unit tests import this module with torch
    # only).
    from vllm_ascend.device.device_op import DeviceOperator

    tail_meta = dataclasses.replace(
        attn_metadata, block_table=attn_metadata.block_table[K:]
    )
    return DeviceOperator.indexer_select_post_process(
        sfa_impl,
        q_li[D:N],
        q_li_scale,
        q_li_shape_ori,
        weights[D:N],
        kv_cache,
        tail_meta,
        actual_seq_lengths_query[K:] - D,
        actual_seq_lengths_key[K:],
        getattr(sfa_impl, "enable_sparse_li_c8", False),
        getattr(sfa_impl, "use_torch_npu_lightning_indexer", False),
    )


def _coarse_screen(
    q_bar: torch.Tensor,
    w_bar: torch.Tensor,
    kv_cache: tuple,
    block_table: torch.Tensor,
    block_size: int,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Torch proxy screen: per-proxy top-4096 over its full prefix.

    Replaces the native npu_lightning_indexer proxy scan so the candidate
    superset is _COARSE_BUDGET (4096, the paper's number) rather than the
    native 2048 sparse_count hard limit. Score formula matches the native
    fp32 indexer: score[r, p] = sum_h w_bar[r, h] * relu(q_bar[r, h] . k[r, p])
    over p in [0, seq_lens[r]) (= L + g, the full prefix; causality is left
    to the SFA attention kernel). NOTE: unlike the native indexer, which
    scores each query row independently with that row's own q/w, this
    screen scores a REQUEST-LEVEL mean proxy (q_bar/w_bar averaged over the
    request's g queries -- the PIVOT paper's design) so each request shares
    one candidate set. Returns [R, _COARSE_BUDGET] 0-based positions, -1
    padded where the prefix is shorter than the budget.
    """
    device = q_bar.device
    R = q_bar.shape[0]
    k_cache = kv_cache[2]  # [B, S, 1, D] BF16 (PA_BSND)
    kc = k_cache.view(-1, k_cache.shape[-1])  # [B*S, D]

    # int32 max() is unsupported on NPU; lift seq_lens to int64 first. The
    # Python int() is a scalar device->CPU sync, acceptable in the eager
    # decode path (no graph capture under --enforce-eager).
    seq_i64 = seq_lens.to(torch.int64)
    L_max = int(seq_i64.max())

    pos = torch.arange(L_max, dtype=torch.int64, device=device)  # [L_max]
    # slot for key position p of request r: block_table[r, p//bs] * bs + p%bs
    slots = block_table[:, pos // block_size] * block_size + pos % block_size
    k_all = kc[slots.reshape(-1)].view(R, L_max, -1)  # [R, L_max, D]

    # Score in fp32 (was bf16 via q_bar.dtype): the native indexer is fp32
    # end to end (Mmad fp32 accumulate -> Fixp NoQuant fp32 writeback), and
    # at production score magnitude a bf16 score collapses adjacent columns
    # (gaps of a few hundred) to one bf16 value, distorting the top-4096
    # candidate set -- same bug class as the old bf16 _refine_topk (see
    # memory: indexer-refine-tie-root-cause). R is the decode batch request
    # count (small), so a single bmm is fine memory-wise.
    q32 = q_bar.to(torch.float32)  # [R, H, Dh]
    w32 = w_bar.to(torch.float32)  # [R, H]
    k32 = k_all.to(torch.float32)  # [R, L_max, Dh]
    score = torch.relu(torch.bmm(q32, k32.transpose(1, 2)))  # [R, H, L_max]
    score = (score * w32.unsqueeze(-1)).sum(dim=1)  # [R, L_max]

    beyond = pos.unsqueeze(0) >= seq_i64.unsqueeze(1)  # [R, L_max]
    score = score.masked_fill(beyond, float("-inf"))

    if L_max < _COARSE_BUDGET:
        pad = score.new_full((R, _COARSE_BUDGET - L_max), float("-inf"))
        score = torch.cat([score, pad], dim=-1)  # [R, _COARSE_BUDGET]

    vals, cols = torch.topk(score, _COARSE_BUDGET, dim=-1)  # [R, _COARSE_BUDGET]
    cols = cols.masked_fill(vals == float("-inf"), -1)
    return cols  # [R, _COARSE_BUDGET], 0-based, -1 padded


def _inject_local_window(
    C: torch.Tensor,
    seq_lens: torch.Tensor,
    g: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append each request's decode local-window union to its candidate row.

    Paper Appendix B (decode variant): each query refines over (pool U its
    own W_t = [t-W+1, t]) with duplicates removed, W >= g so the window
    covers every token generated within the step. The op refines over a
    SHARED per-request candidate row, so the per-query windows enter the row
    as their group union [L-g+1, L+g) -- which for W = g is exactly the union
    of the g queries' windows (query row t's window is [t-g+1, t]; the union
    over t in [L, L+g) is [L-g+1, L+g)). Each query row still sees its own
    window in the domain; entries outside the row's causal prefix are
    masked by the SFA kernel downstream, so the widened union is harmless
    per row. Entries compete by score in the refine -- the paper's
    semantics, not a forced-in reserve slot.

    Mechanics. The union is deduplicated against C (the pool is a top-k of
    positions, so recent ones are already in it) and the genuinely-new
    entries are appended, widening the row from _COARSE_BUDGET to at most
    _COARSE_BUDGET + 2g - 1 (the op imposes no candidate-width bound: tiling
    reads s2Size = candidates.shape[1], workspace scales linearly). The row
    is then COMPACTED -- valid candidates gathered to the front, -1 pad to
    the tail. Compaction is required, not cosmetic: the op's S2 walk covers
    candidate columns [0, aslk) and treats every walked column as a real
    candidate, so C's own short-prefix -1 padding (columns [valid_C, c))
    must NOT sit between the coarse prefix and the appended window -- an
    aslk of L+g would stop at the padding (the window's columns live past c
    and are never walked: own tokens lost) and an aslk past the padding
    walks -1 slots. Front-loading makes the whole valid domain reachable.

    Returns (C', aslk'): C' [R, c'] int64 (dtype unchanged) whose valid
    entries are unique, < L+g, and exactly the columns [0, aslk'[r]) --
    nothing else. aslk' = the row's valid candidate count (op contract:
    per-request S2 bound), dtype matching seq_lens. In the lossless region
    (L + g <= _COARSE_BUDGET) _coarse_screen already returned the whole
    prefix [0, L) and only the g own tokens are new, so C' holds exactly the
    whole prefix [0, L+g) with aslk' = L+g -- the same candidate set the
    native indexer scans -- and refine top-k keeps the lossless contract
    bit-identically.
    """
    device = C.device
    R, c = C.shape
    Lg = seq_lens.to(torch.int64)  # [R] = L + g per request
    L = Lg - g  # [R] prefix length BEFORE this step's own tokens
    aslk_dtype = seq_lens.dtype

    # Window union [L-g+1, L+g): the g query rows' own windows joined. Query
    # row t (t in [L, L+g)) attends up to t, so its W_t = [t-g+1, t]; the
    # union over the group is [L-g+1, L+g). Positions < 0 are dropped via
    # the `valid` mask below.
    win = torch.arange(-(g - 1), g, device=device)  # [2g-1]
    win = L.view(R, 1) + win.view(1, -1)  # [R, 2g-1] absolute positions
    valid = (win >= 0) & (win < Lg.view(R, 1))

    # Dedup against C: C's valid entries are unique positions, so membership
    # of each window position in the request's row is an elementwise
    # broadcast compare + any-reduce -- no sort, no index search. Exactness:
    # the row's -1 padding never equals a valid (>= 0) window entry, and
    # `valid` already drops the window's own negative positions, so
    # `present` is exactly set-membership. (Alternative bitmap membership in
    # git history; broadcast-compare measured fastest on CPU.)
    present = (C[:, None, :] == win[:, :, None]).any(dim=2)  # [R, 2g-1]
    new = valid & ~present  # [R, 2g-1] window entries not already in C

    max_new = int(new.sum(dim=1).max()) if R else 0
    if max_new == 0:
        # Nothing to add: the lossless region cannot reach here (the g own
        # tokens [L, L+g) never enter a [0, L)-domain pool, so every row
        # adds at least one), so this is the truncated region with the
        # window already covered by C -- clamp-only aslk over C's compact
        # rows (valid front, -1 tail) is exact.
        return C, torch.clamp(seq_lens, max=c)

    # Append the genuinely-new entries in natural ascending position order.
    # The old key=new<<40 - win descending argsort actually yields ASCENDING
    # positions (smaller win => larger key => ranked first; verified bitwise
    # against the live pipeline), and win is ascending by construction -- so
    # new entries need no sort at all, they write by their running index.
    W = c + max_new
    out = C.new_full((R, W), -1)
    out[:, :c] = C  # C's own valid entries already sit at the row front
    valid_C = (C >= 0).sum(dim=1)  # [R] C's valid prefix length

    # Compact via a micro-scatter: only the <= 2g-1 new columns per row move.
    # dst = valid_C + running-new-index is consecutive (no collisions) and
    # stays in [valid_C, W); non-new window columns write -1 to the dead pad
    # column W-1 (W-1 >= c here, so C's copy is never overwritten). Order is
    # irrelevant to scoring -- only the candidate SET matters.
    new_idx = torch.cumsum(new.to(torch.int32), dim=1) - 1  # 0..new_count-1
    dst = valid_C[:, None] + new_idx  # [R, 2g-1] target column per entry
    out.scatter_(1, torch.where(new, dst, W - 1), torch.where(new, win, -1))
    aslk_out = (valid_C + new.sum(dim=1)).to(aslk_dtype)  # [R] valid count
    return out, aslk_out


def _refine_topk(
    q_dq: torch.Tensor,
    weights: torch.Tensor,
    C: torch.Tensor,
    req_ids: torch.Tensor,
    kv_cache: tuple,
    block_table: torch.Tensor,
    block_size: int,
    N: int,
) -> torch.Tensor:
    """Per-query top-k over the broadcast candidate set, native formula.

    score[n, j] = sum_h w[n, h] * ReLU(q[n, h] . k_cand[req(n), j]) -- the
    exact formula the indexer computes, scored in fp32: bf16 q/k/weights in,
    fp32 MM accumulate and fp32 scale+reduce, mirroring the native
    npu_lightning_indexer / npu_indexer_refine op (Mmad -> fp32 L0C, fp32 GM
    writeback). -1 candidate slots are masked to -inf before top-k (no causal
    mask here: the native indexer scans the full [0, L+g) domain and lets the
    SFA attention kernel apply causality).
    """
    device = q_dq.device
    R, c = C.shape
    Dh = q_dq.shape[2]  # head_dim (NOT the query row count `D` of select_topk)
    k_cache = kv_cache[2]  # [B, S, 1, Dh] BF16 (PA_BSND)
    kc = k_cache.view(-1, k_cache.shape[-1])  # [B*S, Dh]

    r = torch.arange(R, dtype=torch.int64, device=device)
    c_safe = C.clamp(min=0).to(torch.int64)  # -1 slots clamped for gather
    slots_c = (
        block_table[r.unsqueeze(1), c_safe // block_size] * block_size + c_safe % block_size
    )  # [R, c]
    k_cand = kc[slots_c.reshape(-1)].view(R, c, Dh)  # [R, c, Dh]

    # Broadcast candidates to query rows and score in one bmm per chunk:
    # [N, H, Dh] x [N, Dh, c] -> [N, H, c].
    # Score in fp32 (was bf16 via q_dq.dtype). At production score magnitude
    # ~1.9e6 a bf16 buffer's ulp is ~14844 while the adjacent-column gap is a
    # few hundred -> ~56 columns collapse to one bf16 value and the top-k is
    # destroyed; fp32 keeps them distinct and matches the fp32 op (see
    # memory: indexer-refine-tie-root-cause).
    q32 = q_dq.to(torch.float32)
    w32 = weights.to(torch.float32)
    k32 = k_cand.to(torch.float32)
    C_n = C[req_ids]  # [N, c]
    score = torch.empty(N, c, dtype=torch.float32, device=device)
    chunk = 256
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        att = torch.relu(
            torch.bmm(q32[s:e], k32[req_ids[s:e]].transpose(1, 2))
        )  # [chunk, H, c]
        score[s:e] = (att * w32[s:e].unsqueeze(-1)).sum(dim=1)

    invalid = C_n < 0
    score = score.masked_fill(invalid, float("-inf"))
    vals, cols = torch.topk(score, _REFINE_BUDGET, dim=-1)
    true_pos = C_n.gather(1, cols)
    true_pos = true_pos.masked_fill(vals == float("-inf"), -1)

    return true_pos.view(N, 1, _REFINE_BUDGET).to(torch.int32)


def _report(
    seq_lens: torch.Tensor,
    counts: torch.Tensor,
    C: torch.Tensor,
    topk_indices: torch.Tensor,
    D: int,
    K: int,
    g: int,
    R_all: int,
    N: int,
    aslk: torch.Tensor | None = None,
) -> None:
    """Traceable prints to localize any precision problem on the real op.

    Three checks, each narrowing the failure location:
      1. entry shape (segment + whole-batch geometry),
      2. refine-domain contract (per-row valid candidate count == the aslk the
         op consumed -- C is post-local-window, so it can legitimately carry
         up to 2g-1 more columns than _COARSE_BUDGET; the walk bound is the
         ground truth, not the coarse width) and out-of-range positions,
      3. lossless self-check (decode rows with L+g <= 2048 must reproduce
         the full prefix, i.e. the exact dense key set).
    """
    seq = seq_lens.to(torch.int64)
    sl_min, sl_max = int(seq.min()), int(seq.max())
    # counts (per-request query tokens) expose the real batch layout under
    # concurrency: a non-uniform batch is what cross-bleeds request keys.
    counts_list = [int(c) for c in counts]
    logger.info(
        "PIVOT[dbg] entry: rows=%d/%d reqs=%d/%d g=%d budget=%d seq_lens=[%d..%d] counts=%s",
        D, N, K, R_all, g, _COARSE_BUDGET, sl_min, sl_max, counts_list,
    )

    # Refine-domain contract: per-row valid candidate count must equal the
    # aslk the op consumed (the S2 walk bound), and no candidate position may
    # reach past its request's L+g. C is post-local-window, so its valid count
    # can legitimately exceed _COARSE_BUDGET by up to 2g-1 -- the aslk value
    # is the ground truth. Keep int64: NPU max()/clamp() reject int32.
    C_i64 = C.to(torch.int64)
    valid = C_i64 >= 0
    valid_per_row = valid.sum(-1)  # [K]
    if aslk is not None:
        mismatch = (valid_per_row != aslk.to(torch.int64)).sum()
        expected_lo = int(aslk.min())
        expected_hi = int(aslk.max())
    else:
        mismatch = -1  # unknown -> report range only
        expected_lo, expected_hi = -1, -1
    out_of_range = (C_i64.clamp(min=0).max(-1).values >= seq).sum()
    logger.info(
        "PIVOT[dbg] refine-domain contract: valid_per_row=[%d..%d] aslk=[%d..%d] "
        "mismatch_rows=%d out_of_range_rows=%d",
        int(valid_per_row.min()), int(valid_per_row.max()),
        expected_lo, expected_hi,
        int(mismatch), int(out_of_range),
    )

    # Lossless self-check: rows with L+g <= 2048 must equal [0..L+g) exactly.
    out = topk_indices[:D, 0].to(torch.int64)
    violations = 0
    first = None
    for r in range(K):
        Lg = int(seq[r])
        if Lg > _REFINE_BUDGET:
            continue
        for i in range(g):
            n = r * g + i
            got = sorted(int(v) for v in out[n] if v >= 0)
            want = list(range(Lg))
            if got != want:
                violations += 1
                if first is None:
                    first = (n, Lg, got[:16])
    if violations:
        logger.warning(
            "PIVOT[dbg] LOSSLESS VIOLATION: %d rows (of %d in the lossless "
            "regime) are not [0..L+g); first row=%d L+g=%d got=%s",
            violations, D, first[0], first[1], first[2],
        )
    else:
        logger.info(
            "PIVOT[dbg] lossless self-check: PASS (%d rows, L+g<=%d)",
            D, _REFINE_BUDGET,
        )
