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
        handle_tail: bool = True,
    ) -> torch.Tensor | None:
        """Return decode-head topk_indices (0-based logical key positions).

        Decode requests sit at the head of the batch (the engine reorders to
        decode -> ... -> prefill), so the leading run of requests with a
        uniform query count g is the decode segment: requests [0, K), query
        rows [0, D). That segment runs through PIVOT; the prefill tail is
        handled by the caller:
          - handle_tail=True (default): the tail runs through the NATIVE
            indexer and the results are concatenated, then the full-batch
            graph/index-cache guards are applied (legacy behavior, used when
            prefill-PIVOT is off);
          - handle_tail=False (prefill-PIVOT on): return the decode head raw
            [0, D) with NO tail and NO full-batch guards -- the caller (the
            three-way dispatch) owns the tail (via select_topk_prefill) and
            the guards (via _apply_output_guards on the combined result).

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

        # Decode segment: K requests / D rows are the engine's authoritative
        # decode/prefill split (utils.split_decodes_and_prefills reorders
        # decodes to the batch head). Do not re-derive them from the
        # cumulative geometry -- the old leading-run inference (first request
        # with count != g) could misfire when a prefill request carries
        # exactly g rows and extend K past the real decode head. g (the MTP
        # step-group size) is the one runtime fact not in the metadata; it is
        # the head request's row count (decode-at-head).
        K = attn_metadata.num_decodes
        D = attn_metadata.num_decode_tokens
        g = int(cum[0])
        if not 2 <= g <= _MAX_GROUP:
            # Not a grouped-decode head (single-token decode, or a prefill
            # leading the batch): nothing to amortize.
            logger.warning_once(
                "PIVOT: leading request carries g=%d query rows (not a "
                "grouped-decode head); bailing the whole batch to the native "
                "indexer. If this fires in PD co-location, the decode->prefill "
                "reorder did not put decodes at the batch head.", g)
            return None

        # O(1) geometry self-check: the decode head must be exactly K requests
        # of g rows each, ending at the metadata row boundary. The old O(R)
        # counts==g scan guaranteed this by construction; metadata K/D do not,
        # so assert it here -- preserves the bail-to-native on any
        # decode-not-at-head / non-uniform geometry.
        if K * g != D or int(cum[K - 1]) != D:
            logger.warning_once(
                "PIVOT: decode head geometry (K=%d g=%d D=%d) inconsistent "
                "with metadata num_decodes/num_decode_tokens; bailing the "
                "whole batch to the native indexer.", K, g, D)
            return None

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
            C, aslk_op = _inject_local_window(
                C, seq_lens[:K] - g, seq_lens[:K], g)
        else:
            # aslk drives the per-request S2 chunk loop over the CANDIDATE
            # LIST (row width = C.shape[1]). In the truncated region
            # L+g > 4096 an unclamped aslk makes the kernel read candidate
            # columns past the row end (garbage/adjacent rows); clamp.
            aslk_op = torch.clamp(seq_lens[:K], max=C.shape[1])

        # ---- 3. refine: broadcast C, score, top-k -------------------------
        # Query row -> request id within the decode segment. The decode head
        # is validated uniform-g above (K*g == D), so each request owns g
        # rows: repeat_interleave(arange(K), g) == the old counts[:K] form.
        req_ids = torch.repeat_interleave(
            torch.arange(K, dtype=torch.int64, device=device), g
        )  # [D]
        if envs.VLLM_ASCEND_PIVOT_REFINE_USE_OP:
            # Validated op (NPU 9/9 tie-aware PASS). The op groups the
            # per-request candidates C via the cumulative aslq internally.
            # NO try/except: an op failure now propagates loudly instead of
            # silently recomputing via torch -- the fallback hid which path
            # actually ran (debug instrumentation; restore the fallback once
            # the window/prefill geometry is re-validated).
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
                # Tiling hard-requires int32 lengths; aslk_op is int64
                # once the local window (_inject_local_window) is on --
                # without the cast the op fails tiling and this path
                # silently falls back to the torch reference.
                actual_seq_lengths_query=cum[:K].to(torch.int32),
                actual_seq_lengths_key=aslk_op.to(torch.int32),
                block_table=attn_metadata.block_table[:K],
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=_REFINE_BUDGET,
            )  # [D, 1, _REFINE_BUDGET] int32
            # One-shot confirmation that the op path is live -- without
            # this a silently failing op would make gsm8k runs
            # indistinguishable from USE_OP=0.
            logger.info_once(
                "PIVOT refine: using npu_indexer_refine op "
                "(VLLM_ASCEND_PIVOT_REFINE_USE_OP=1).")
            # Dump is diagnostics: guarded so a dump bug can never read
            # "op failed" or break the decode path. _dump_real_inputs
            # self-protects too.
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

        if handle_tail and K != R_all:
            # Mixed batch, native tail (fallback, prefill-PIVOT off): the
            # prefill tail (requests [K, R_all), rows [D, N)) stays on the
            # native indexer.
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
        else:
            # Pure decode head (K == R_all), or the caller owns the tail and
            # the full-batch guards (handle_tail=False): return the decode
            # head raw.
            topk_indices = topk_dec

        if handle_tail:
            # Full-batch guards (graph row pad + index-cache width pad). With
            # handle_tail=False these are applied by the caller on the
            # combined decode+prefill result instead.
            topk_indices = _apply_output_guards(
                topk_indices, sfa_impl, N_in, N)

        if _ENABLE_REPORT and not _capturing():
            try:
                _report(seq_lens[:K], [g] * K, C, topk_indices, D, K, g,
                        R_all, N, aslk=aslk_op)
            except Exception as e:  # diagnostics must never take down the decode path
                logger.warning("PIVOT[dbg] _report failed: %s", e)

        logger.debug(
            "PIVOT refine: rows=%d/%d reqs=%d/%d g=%d", D, N, K, R_all, g
        )
        return topk_indices


    @staticmethod
    def select_topk_prefill(
        sfa_impl,
        q_li: torch.Tensor,
        q_li_scale: torch.Tensor | None,
        q_li_shape_ori: tuple | None,
        weights: torch.Tensor,
        kv_cache: tuple,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        prefill_tail_start: int,
    ) -> torch.Tensor | None:
        """Return prefill-tail topk_indices [N-D, 1, 2048] via prefill-PIVOT.

        The prefill tail is the TND rows [D, N) = the requests whose cumulative
        query ends exceed the decode-head row count D = prefill_tail_start
        (= num_decode_tokens). Each prefill request's q_r query rows are split
        into positional groups of g (paper Eq. 2, last group may be smaller);
        groups become the R axis, so the whole decode PIVOT pipeline (mean
        proxy -> coarse screen -> per-group window -> refine) runs unchanged
        with R = P groups instead of R = K requests.

        Coarse domain: [0, group_start) per group -- the pool is recalled at
        the group's FIRST position (paper Eq. 5), so a group's own tokens
        (>= group_start) never enter the pool (the window supplies them) but
        EARLIER groups' own tokens DO (they are causal for this group).
        Combined with the window [group_start-g+1, group_end), a full group's
        refine domain is the complete causal prefix [0, group_end). This is
        the decode formula [0, L) + [L-g+1, L+g) = [0, L+g) generalized to
        group index j: [0, group_start_j) + [group_start_j-g+1, group_end_j).
        (A per-request [0, L_r) domain -- prefix before the whole chunk --
        would drop every earlier group's own tokens from a later group's
        candidate set, a correctness gap for multi-group requests.)

        Returns None when prefill-PIVOT is inapplicable (C8, g out of range);
        the caller then falls the whole batch to the native indexer. All
        geometry is pure tensor ops; the few int()/bool() reductions are the
        same lossless-gate / count decisions select_topk already makes in the
        eager decode path (prefill is always eager, never in a graph).
        """
        if getattr(sfa_impl, "enable_sparse_li_c8", False):
            logger.warning_once(
                "PIVOT prefill: enable_sparse_li_c8 is set; falling back to "
                "the native indexer (PIVOT supports the BF16 path only).")
            return None

        g = envs.VLLM_ASCEND_PIVOT_PREFILL_GROUP
        if not 2 <= g <= _MAX_GROUP:
            logger.warning_once(
                "PIVOT prefill: PREFILL_GROUP=%d out of [2, %d]; falling back "
                "to the native indexer.", g, _MAX_GROUP)
            return None

        cum = actual_seq_lengths_query  # [R_all] cumulative query ends
        seq_lens = actual_seq_lengths_key  # [R_all] == L_r + q_r
        if cum.shape[0] == 0:
            return None
        N = attn_metadata.num_actual_tokens
        device = q_li.device
        D = prefill_tail_start
        block_size = attn_metadata.block_size

        # ---- prefill segment: requests with rows [D, N) -------------------
        # Decode requests (rows < D) carry g query rows each; prefill requests
        # are those whose cumulative end > D (the last decode request ends
        # exactly at D, so `cum > D` excludes it).
        pre_req = cum > D                      # [R_all] bool
        R_pre = int(pre_req.sum())
        if R_pre == 0:
            return None
        tail_cum = cum[pre_req] - D            # [R_pre], rebased: first == q_0
        tail_seq = seq_lens[pre_req]           # [R_pre]
        tail_bt = attn_metadata.block_table[pre_req]  # [R_pre, MAX_BLK]
        N_tail = int(tail_cum[-1])             # == total prefill rows (N - D)
        tail_q = _request_counts(tail_cum)     # [R_pre] per-request query counts

        # ---- positional groups (paper Eq. 2) ------------------------------
        n_r = (tail_q + g - 1) // g            # [R_pre] groups per request
        P = int(n_r.sum())
        req_base = torch.cat(
            [torch.zeros(1, dtype=n_r.dtype, device=device), n_r.cumsum(0)[:-1]]
        )                                       # [R_pre] group start idx per req
        # Per-query-row: its position within its request and its group.
        row_start = torch.cat(
            [torch.zeros(1, dtype=tail_cum.dtype, device=device),
             tail_cum[:-1]]
        )                                       # [R_pre] first row of each req
        row_in_req = torch.arange(N_tail, device=device) \
            - torch.repeat_interleave(row_start, tail_q)  # [N_tail]
        group_local = row_in_req // g           # [N_tail] group idx within req
        group_ids = torch.repeat_interleave(req_base, tail_q) \
            + group_local                       # [N_tail] per-row group idx

        # Group sizes: g everywhere except each request's last group.
        last_size = tail_q - (n_r - 1) * g      # [R_pre] last-group size (1..g)
        last_gidx = req_base + n_r - 1          # [R_pre] index of each last group
        group_sizes = torch.full((P,), g, dtype=tail_q.dtype, device=device)
        group_sizes[last_gidx] = last_size
        aslq_refine = group_sizes.cumsum(0)     # [P] cumulative; [-1] == N_tail

        # Absolute KV position of each group's own-token block.
        L_r = tail_seq - tail_q                 # [R_pre] request prefix before chunk
        group_j = torch.arange(P, device=device) \
            - torch.repeat_interleave(req_base, n_r)   # [P] local group idx
        group_start = torch.repeat_interleave(L_r, n_r) + group_j * g  # [P]
        group_end = group_start + group_sizes   # [P]

        # Coarse/refine map candidate rows to KV slots via the OWNING
        # request's block table: per-GROUP block table aligned with C [P, ...].
        group_bt = tail_bt.repeat_interleave(n_r, dim=0)  # [P, MAX_BLK]

        # ---- mean proxy (per group) --------------------------------------
        q_dq = q_li[D:N]      # [N_tail, H, Dh] raw BF16
        w_t = weights[D:N]    # [N_tail, H]
        H, Dh = q_dq.shape[1], q_dq.shape[2]
        # Grouped mean proxy. scatter_add_ over the FULL [N_tail, H, Dh]
        # index materializes the expanded gidx internally -- for a large
        # prefill tail (N_tail thousands x H*Dh=16384) that is multi-GiB of
        # index tensor and OOMs next to the resident graph (2026-09-11,
        # prefill node). Tile over the row axis; scatter_add_ accumulates, so
        # the tiled result is bit-identical. prefill is always eager, so the
        # Python loop is free.
        q_bar = q_dq.new_zeros(P, H, Dh)
        w_bar = w_t.new_zeros(P, H)
        _MT = 256
        for s in range(0, N_tail, _MT):
            e = min(s + _MT, N_tail)
            gs = group_ids[s:e]                                  # [tile]
            q_bar.scatter_add_(0,
                               gs.view(-1, 1, 1).expand(e - s, H, Dh),
                               q_dq[s:e])
            w_bar.scatter_add_(0,
                               gs.view(-1, 1).expand(e - s, H),
                               w_t[s:e])
        q_bar = q_bar / group_sizes.view(P, 1, 1)
        w_bar = w_bar / group_sizes.view(P, 1)

        # ---- coarse screen (proxy domain [0, group_start) per group) -----
        if bool((group_start <= _COARSE_BUDGET).all()):
            # Lossless fast path: every prefix [0, group_start) fits the
            # budget, so the whole prefix is the candidate set.
            col = torch.arange(_COARSE_BUDGET, dtype=torch.int64, device=device)
            C = torch.where(col[None, :] < group_start[:, None],
                            col[None, :], -1)
        else:
            C = _coarse_screen(q_bar, w_bar, kv_cache, group_bt,
                               block_size, group_start)

        # ---- per-query local window (compete-by-score, decode semantics) --
        if envs.VLLM_ASCEND_PIVOT_LOCAL_WINDOW:
            C, aslk_op = _inject_local_window(C, group_start, group_end, g)
        else:
            aslk_op = torch.clamp(group_end, max=C.shape[1])

        # ---- refine: query row -> group candidate row ---------------------
        req_ids = group_ids  # [N_tail] each row -> its group's candidate row

        if envs.VLLM_ASCEND_PIVOT_REFINE_USE_OP:
            # NO try/except: an op failure now propagates loudly instead of
            # silently recomputing via torch -- the fallback hid which path
            # actually ran (debug instrumentation; restore when the geometry
            # is re-validated).
            # Op prototype requires candidates DT_INT32; _coarse_screen
            # returns torch.topk indices (int64). Cast only on the op path.
            topk_pre = torch.ops._C_ascend.npu_indexer_refine(
                q_dq, kv_cache[2], w_t, C.to(torch.int32),
                # Tiling (indexer_refine_tiling.cpp) hard-requires int32
                # for both length inputs; the positional-group geometry
                # builds int64 (arange / _request_counts) -> cast only on
                # the op path (the torch reference consumes them as-is).
                actual_seq_lengths_query=aslq_refine.to(torch.int32),
                actual_seq_lengths_key=aslk_op.to(torch.int32),
                block_table=group_bt,
                layout_query="TND", layout_key="PA_BSND",
                sparse_count=_REFINE_BUDGET,
            )  # [N_tail, 1, _REFINE_BUDGET] int32
            # One-shot confirmation the prefill op path is live (decode has
            # its own at the decode call; without this a silent prefill op
            # failure is indistinguishable from USE_OP=0).
            logger.info_once(
                "PIVOT prefill: using npu_indexer_refine op "
                "(VLLM_ASCEND_PIVOT_REFINE_USE_OP=1).")
            # Same caller-side column->position gather as decode: the op
            # emits candidate COLUMN indices, not KV positions.
            # Map op column indices back to KV positions via the group's
            # candidate row C[req_ids]. Materializing C[req_ids] at once is
            # [N_tail, c'] -- for a large prefill tail (N_tail thousands,
            # c' up to 4096+2g-1) that is hundreds of MiB of intermediate
            # and OOMs next to the resident graph (same class as the
            # coarse-screen R*L_max blowup; 2026-09-11 NPU 3/6). Tile the
            # gather over the row axis; prefill is always eager (never
            # graph-captured) so the Python loop is free.
            _GT = 64
            for s in range(0, N_tail, _GT):
                e = min(s + _GT, N_tail)
                col_s = topk_pre[s:e, 0].to(torch.int64)  # [tile, 2048]
                pos_s = C[req_ids[s:e]].gather(1, col_s.clamp(min=0))  # [tile, 2048]
                topk_pre[s:e, 0] = torch.where(col_s < 0, -1, pos_s).to(torch.int32)
        else:
            topk_pre = _refine_topk(q_dq, w_t, C, req_ids, kv_cache,
                                    group_bt, block_size, N_tail)

        assert int(aslq_refine[-1]) == N_tail, \
            f"PIVOT prefill: group cum [-1]={int(aslq_refine[-1])} != N_tail={N_tail}"

        logger.debug("PIVOT prefill refine: rows=%d reqs=%d groups=%d g=%d",
                     N_tail, R_pre, P, g)
        return topk_pre


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


def _apply_output_guards(
    topk_indices: torch.Tensor,
    sfa_impl,
    N_in: int,
    N: int,
) -> torch.Tensor:
    """Pad PIVOT topk_indices to the native row count and index-cache width.

    Shared by select_topk (handle_tail=True) and the three-way dispatch (on
    the combined decode+prefill result). Two pads, both -1 tails:
      - graph padding: rows [N, N_in) get -1 rows so the output row count
        matches the native path (num_input_tokens);
      - use_index_cache width: pad the width to the buffer width so the read
        side returns a full-width buffer.
    """
    device = topk_indices.device
    if N_in > N:
        row_pad = torch.full(
            (N_in - N, 1, topk_indices.shape[-1]),
            -1,
            dtype=topk_indices.dtype,
            device=device,
        )
        topk_indices = torch.cat([topk_indices, row_pad], dim=0)
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
    return topk_indices


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

    # The dense gather k_all=[R, L_max, D] and the fp32 score=[R, H, L_max]
    # both scale as R*L_max. In the decode path R is the (small) request
    # count, but in the prefill path R is the positional-group count
    # (N_pre/g -- thousands on a long chunk) and L_max is the longest group
    # prefix, so the product OOMs on one shot (real prefill run: a 5.6 GiB
    # gather on top of a 54 GiB resident graph). Each group top-k is over its
    # OWN prefix and the -inf padding is per-row, so tiling over R is
    # bit-identical to the monolithic call while bounding every intermediate
    # by the tile size. _TILE keeps the worst-case fp32 score
    # [tile, H, L_max] within the prefill headroom for L_max up to 8K.
    pos = torch.arange(L_max, dtype=torch.int64, device=device)  # [L_max]
    _TILE = 64
    cols_out = torch.full((R, _COARSE_BUDGET), -1, dtype=torch.int64,
                          device=device)
    for s in range(0, R, _TILE):
        e = min(s + _TILE, R)
        bt_t = block_table[s:e]   # [t, MAX_BLK]
        seq_t = seq_i64[s:e]      # [t]
        # slot for key position p of request r: block_table[r, p//bs]*bs + p%bs
        slots = bt_t[:, pos // block_size] * block_size + pos % block_size
        k_all = kc[slots.reshape(-1)].view(e - s, L_max, -1)  # [t, L_max, D]

        # Score in fp32 (was bf16 via q_bar.dtype): the native indexer is
        # fp32 end to end (Mmad fp32 accumulate -> Fixp NoQuant fp32
        # writeback), and at production score magnitude a bf16 score
        # collapses adjacent columns (gaps of a few hundred) to one bf16
        # value, distorting the top-4096 candidate set -- same bug class as
        # the old bf16 _refine_topk (see memory: indexer-refine-tie-root-cause).
        q32 = q_bar[s:e].to(torch.float32)  # [t, H, Dh]
        w32 = w_bar[s:e].to(torch.float32)  # [t, H]
        k32 = k_all.to(torch.float32)       # [t, L_max, Dh]
        score = torch.relu(torch.bmm(q32, k32.transpose(1, 2)))  # [t, H, L_max]
        score = (score * w32.unsqueeze(-1)).sum(dim=1)           # [t, L_max]

        beyond = pos.unsqueeze(0) >= seq_t.unsqueeze(1)  # [t, L_max]
        score = score.masked_fill(beyond, float("-inf"))

        if L_max < _COARSE_BUDGET:
            pad = score.new_full((e - s, _COARSE_BUDGET - L_max),
                                 float("-inf"))
            score = torch.cat([score, pad], dim=-1)  # [t, _COARSE_BUDGET]

        vals, cols = torch.topk(score, _COARSE_BUDGET, dim=-1)
        cols_out[s:e] = cols.masked_fill(vals == float("-inf"), -1)
    return cols_out  # [R, _COARSE_BUDGET], 0-based, -1 padded


def _inject_local_window(
    C: torch.Tensor,
    win_base: torch.Tensor,
    end: torch.Tensor,
    g: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append each group's local-window union to its candidate row.

    Decode variant (caller passes win_base=seq_lens-g, end=seq_lens): each
    query refines over (pool U its own W_t = [t-W+1, t]) with duplicates
    removed, W >= g so the window covers every token generated within the
    step. The op refines over a SHARED per-group candidate row, so the
    per-query windows enter the row as their group union
    [win_base-g+1, end) -- which for W = g is exactly the union of the g
    queries' windows (query row t's window is [t-g+1, t]; the union over
    t in [win_base, win_base+g) is [win_base-g+1, win_base+g) =
    [win_base-g+1, end)). Prefill variant (caller passes
    win_base=group_start, end=group_end): a positional group's own tokens
    [group_start, group_end) plus the g-1 preceding, i.e.
    [group_start-g+1, group_end) -- the same [win_base-g+1, end) shape, so a
    single body serves both stages. Each query row still sees its own window
    in the domain; entries outside the row's causal prefix are masked by the
    SFA kernel downstream, so the widened union is harmless per row. Entries
    compete by score in the refine -- the paper's semantics, not a forced-in
    reserve slot.

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
    entries are unique, < end, and exactly the columns [0, aslk'[r]) --
    nothing else. aslk' = the row's valid candidate count (op contract:
    per-group S2 bound), dtype matching win_base. In the lossless region
    (end <= _COARSE_BUDGET) _coarse_screen already returned the whole prefix
    [0, win_base) and only the own tokens are new, so C' holds exactly the
    whole prefix [0, end) with aslk' = end -- the same candidate set the
    native indexer scans -- and refine top-k keeps the lossless contract
    bit-identically.
    """
    device = C.device
    R, c = C.shape
    win_base = win_base.to(torch.int64)  # [R] absolute own-token block start
    end = end.to(torch.int64)            # [R] absolute window end (exclusive)
    aslk_dtype = win_base.dtype

    # Window union [win_base-g+1, end): the group's own tokens plus the g-1
    # lookback. Positions < 0 are dropped via the `valid` mask below.
    win = torch.arange(-(g - 1), g, device=device)  # [2g-1]
    win = win_base.view(R, 1) + win.view(1, -1)  # [R, 2g-1] absolute positions
    valid = (win >= 0) & (win < end.view(R, 1))

    # Dedup against C: C's valid entries are unique positions, so membership
    # of each window position in the request's row is an elementwise
    # broadcast compare + any-reduce -- no sort, no index search. Exactness:
    # the row's -1 padding never equals a valid (>= 0) window entry, and
    # `valid` already drops the window's own negative positions, so
    # `present` is exactly set-membership. (Alternative bitmap membership in
    # git history; broadcast-compare measured fastest on CPU.)
    # Membership broadcast (C == win).any over the c columns materializes an
    # intermediate [R, 2g-1, c] bool -- for a large prefill (R = P groups,
    # c = 4096+) that is hundreds of MiB and OOMs next to the resident graph
    # (2026-09-11). Tile over the row axis: the per-tile intermediate is
    # [tile, 2g-1, c] (~2 MiB at tile=64, c=4096). decode (R = K, small) runs
    # in a single tile, behavior unchanged.
    present = torch.zeros(R, win.shape[1], dtype=torch.bool, device=device)
    _WT = 64
    for s in range(0, R, _WT):
        e = min(s + _WT, R)
        present[s:e] = (C[s:e, None, :] == win[s:e, :, None]).any(dim=2)
    new = valid & ~present  # [R, 2g-1] window entries not already in C

    max_new = int(new.sum(dim=1).max()) if R else 0
    if max_new == 0:
        # Nothing to add: the lossless region cannot reach here (the g own
        # tokens [L, L+g) never enter a [0, L)-domain pool, so every row
        # adds at least one), so this is the truncated region with the
        # window already covered by C -- clamp-only aslk over C's compact
        # rows (valid front, -1 tail) is exact.
        return C, torch.clamp(end, max=c)

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
    # dst = valid_C + running-new-index is UNIQUE per row (the cumsum over new
    # entries is 1:1), so every write targets a distinct column -- no duplicate
    # index writes. The previous version also wrote -1 to the dead pad column
    # W-1 for non-new columns; when a row's new_count hit max_new, that dead
    # column coincided with the last new entry's target (win == L+g-1, the
    # newest own token), producing duplicate-index writes whose order torch's
    # scatter_ leaves unspecified. On CPU the row-major order happened to let
    # the last column (always new=True) win, so bitwise tests passed; on NPU
    # torch_npu the collision is a real race -- the newest own token could be
    # overwritten with -1 and silently dropped from the refine domain. Writing
    # ONLY the new entries (out is pre-filled with -1) removes the collision.
    new_idx = torch.cumsum(new.to(torch.int32), dim=1) - 1  # 0..new_count-1
    dst = valid_C[:, None] + new_idx  # [R, 2g-1] target column per entry
    nz_r, nz_c = new.nonzero(as_tuple=True)
    if nz_r.numel():
        out[nz_r, dst[nz_r, nz_c]] = win[nz_r, nz_c]
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
    # The bmm output [chunk, H, c] fp32 must fit the device headroom: on a
    # near-full NPU (58/61 GiB resident, ~116 MiB free) chunk=256's 130 MiB
    # output OOMs the torch reference (USE_OP=0 / dump golden). chunk=64
    # drops it to ~33 MiB; the loop is row-parallel slicing so results are
    # unchanged. (k_cand above is still full [R, c, Dh] -- escalate to
    # c-blocking if a large group count R OOMs there first.)
    chunk = 64
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
