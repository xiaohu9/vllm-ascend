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
            return None

        counts = _request_counts(cum)  # [R]
        g = int(counts[0])
        if not 2 <= g <= _MAX_GROUP:
            # Not a grouped-decode head (single-token decode, or a prefill
            # leading the batch): nothing to amortize.
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
        # formula (sum_h w_bar * relu(q_bar . k)) over the full [0, L+g)
        # prefix; the SFA attention kernel applies causality downstream.
        C = _coarse_screen(
            q_bar, w_bar, kv_cache, attn_metadata.block_table[:K],
            attn_metadata.block_size, seq_lens[:K],
        )

        # ---- 3. refine: broadcast C, score, top-k -------------------------
        # Query row -> request id within the decode segment (the leading run
        # is uniform, so this equals repeat_interleave(arange(K), g)).
        req_ids = torch.repeat_interleave(
            torch.arange(K, dtype=torch.int64, device=device), counts[:K]
        )  # [D]
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
                _report(seq_lens[:K], counts[:K], C, topk_indices, D, K, g, R_all, N)
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
    native 2048 sparse_count hard limit. Score formula matches the BF16
    indexer: score[r, p] = sum_h w_bar[r, h] * relu(q_bar[r, h] . k[r, p])
    over p in [0, seq_lens[r]) (= L + g, the full prefix; causality is left
    to the SFA attention kernel). Returns [R, _COARSE_BUDGET] 0-based
    positions, -1 padded where the prefix is shorter than the budget.
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

    score = torch.relu(torch.bmm(q_bar, k_all.transpose(1, 2)))  # [R, H, L_max]
    score = (score * w_bar.unsqueeze(-1)).sum(dim=1)  # [R, L_max]

    beyond = pos.unsqueeze(0) >= seq_i64.unsqueeze(1)  # [R, L_max]
    score = score.masked_fill(beyond, float("-inf"))

    if L_max < _COARSE_BUDGET:
        pad = score.new_full((R, _COARSE_BUDGET - L_max), float("-inf"))
        score = torch.cat([score, pad], dim=-1)  # [R, _COARSE_BUDGET]

    vals, cols = torch.topk(score, _COARSE_BUDGET, dim=-1)  # [R, _COARSE_BUDGET]
    cols = cols.masked_fill(vals == float("-inf"), -1)
    return cols  # [R, _COARSE_BUDGET], 0-based, -1 padded


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
    exact formula the BF16 indexer computes. -1 candidate slots are masked to
    -inf before top-k (no causal mask here: the native indexer scans the full
    [0, L+g) domain and lets the SFA attention kernel apply causality).
    """
    device = q_dq.device
    R, c = C.shape
    D = q_dq.shape[2]
    k_cache = kv_cache[2]  # [B, S, 1, D] BF16 (PA_BSND)
    kc = k_cache.view(-1, k_cache.shape[-1])  # [B*S, D]

    r = torch.arange(R, dtype=torch.int64, device=device)
    c_safe = C.clamp(min=0).to(torch.int64)  # -1 slots clamped for gather
    slots_c = (
        block_table[r.unsqueeze(1), c_safe // block_size] * block_size + c_safe % block_size
    )  # [R, c]
    k_cand = kc[slots_c.reshape(-1)].view(R, c, D)  # [R, c, D]

    # Broadcast candidates to query rows and score in one bmm per chunk:
    # [N, H, D] x [N, D, c] -> [N, H, c].
    C_n = C[req_ids]  # [N, c]
    score = torch.empty(N, c, dtype=q_dq.dtype, device=device)
    chunk = 256
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        att = torch.relu(
            torch.bmm(q_dq[s:e], k_cand[req_ids[s:e]].transpose(1, 2))
        )  # [chunk, H, c]
        score[s:e] = (att * weights[s:e].unsqueeze(-1)).sum(dim=1)

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
) -> None:
    """Traceable prints to localize any precision problem on the real op.

    Three checks, each narrowing the failure location:
      1. entry shape (segment + whole-batch geometry),
      2. proxy-scan contract (the coarse screen's per-row valid count vs
         expected, 4096 wide),
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

    # Proxy-scan contract: valid per-row counts vs expected, out-of-range.
    # Keep int64 here: NPU max()/clamp() reject int32 (DT_INT32 is not in
    # aclnnMaxDim's supported list). torch.topk already returns int64, so the
    # .to() is a defensive no-op.
    C_i64 = C.to(torch.int64)
    valid = C_i64 >= 0
    valid_per_row = valid.sum(-1)  # [K]
    expected = torch.clamp(seq, max=_COARSE_BUDGET)
    mismatch = (valid_per_row != expected).sum()
    out_of_range = (C_i64.clamp(min=0).max(-1).values >= seq).sum()
    logger.info(
        "PIVOT[dbg] proxy-scan contract: valid_per_row=[%d..%d] expected=[%d..%d] "
        "mismatch_rows=%d out_of_range_rows=%d",
        int(valid_per_row.min()), int(valid_per_row.max()),
        int(expected.min()), int(expected.max()),
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
