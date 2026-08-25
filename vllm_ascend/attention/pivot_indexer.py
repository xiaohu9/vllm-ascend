# SPDX-License-Identifier: Apache-2.0
"""PIVOT-Refine indexer for the native SFA decode path.

Implements the PIVOT-Refine scheme (arXiv:2607.24593) as a drop-in
replacement of the per-query full-prefix indexer scan on MTP decode
batches:

  1. mean-proxy: per-request mean of the g = 1+d indexer queries.
  2. proxy scan: one ``npu_quant_lightning_indexer`` call with R proxy
     queries (C width = budget - W, so C∪W equals the 2048 budget; the
     scan key length is ``seq_lens - counts``, i.e. the C domain never
     contains this step's MTP keys).
  3. torch refine: joint C∪W_t scoring per query (same formula and
     dequant domain as the operator) followed by a joint top-k.

Derived quantities are NOT env knobs (user decision, doc v3.4.6): the
window W = g (draft_num + 1) and the C∪W budget = 2048 (native op
limit) are fixed here; only the enable switch and the output width k
live in envs.

All per-step group metadata (counts / group_start / req_ids /
positions_q / window_pos / proxy key lens) is computed ONCE in
``AscendSFAMetadata._build`` for decode states and read here as
fields -- every MLA layer's forward calls into this module, so
anything derivable from metadata must not be recomputed per layer
(user decision, doc v3.4.6).

Design doc:
  docs/source/developer_guide/Design_Documents/pivot_indexer.md (v3.4.6)
"""

import torch
import torch_npu

from vllm.logger import logger
from vllm_ascend import envs

# C∪W total budget. 2048 is the native indexer hard limit
# (SPARSE_LIMIT / TOPK_MAX_SIZE in tiling.cpp), so C width = 2048 - g.
_CANDIDATE_BUDGET = 2048

# Per-request chunk size for the refine C-side scoring (bounds the
# [chunk, g*H, c] bmm temporary). Module constant on purpose: chunking
# granularity is an implementation detail, not a user knob.
_REFINE_CHUNK = 256


class PivotIndexer:
    """PIVOT-Refine top-k selection (V1 torch pass-through, C8 only)."""

    @staticmethod
    def select_topk(
        sfa_impl,
        q_li: torch.Tensor,
        q_li_scale: torch.Tensor,
        q_li_shape_ori: tuple,
        weights: torch.Tensor,
        kv_cache: tuple,
        attn_metadata,
    ) -> torch.Tensor | None:
        """Return topk_indices [N, 1, k] (0-based logical key positions).

        Called from ``AscendSFAImpl.indexer_select_post_process`` when the
        PIVOT switch is on and the batch is a grouped MTP decode batch.
        Returns None for ungrouped batches (g < 2); the caller then falls
        back to the native indexer path.
        """
        if not sfa_impl.enable_sparse_li_c8:
            # V1 targets the C8 indexer path (GLM-5.2-w4a4c8). The BF16
            # path keeps the native indexer.
            raise NotImplementedError(
                "PIVOT-Refine V1 supports enable_sparse_li_c8 only."
            )

        N_in, H, D = q_li_shape_ori
        # Real tokens only: graph batches pad num_input_tokens beyond
        # num_actual_tokens; padded rows get -1 output tails below.
        N = attn_metadata.num_actual_tokens
        R = attn_metadata.seq_lens.shape[0]
        # Group metadata derived once in AscendSFAMetadata._build (decode
        # states only); see the doc §6 sfa_v1 row for the construction.
        counts = attn_metadata.pivot_counts  # [R], == g in steady state
        group_start = attn_metadata.pivot_group_start  # [R]
        req_ids = attn_metadata.pivot_req_ids  # [N]
        positions_q = attn_metadata.pivot_positions_q  # [N]
        window_pos = attn_metadata.pivot_window_pos  # [N, W], -1 out of range
        proxy_key_lens = attn_metadata.pivot_proxy_key_lens  # [R] = seq_lens - counts
        if counts is None or req_ids is None or positions_q is None:
            raise RuntimeError(
                "PIVOT: group metadata fields are missing on "
                "AscendSFAMetadata (expected decode states only)."
            )
        if positions_q.shape[0] != N or window_pos.shape[0] != N:
            raise RuntimeError(
                f"PIVOT: metadata/N mismatch, N={N}, "
                f"positions_q={positions_q.shape[0]}, "
                f"window_pos={window_pos.shape[0]}."
            )
        if R == 0 or N < R:
            raise RuntimeError(f"PIVOT: invalid N={N}, R={R}.")
        g = N // R
        if g < 2:
            # Ungrouped batch (plain decode, 1 query per request): PIVOT has
            # nothing to amortize. Signal the caller to fall back to the
            # native indexer path.
            return None

        k = envs.VLLM_ASCEND_PIVOT_TOPK
        # W and the budget are derived, not configured (doc v3.4.6).
        W = window_pos.shape[1]
        if not (g <= W < k <= _CANDIDATE_BUDGET - W):
            raise RuntimeError(
                f"PIVOT: need g <= W < k <= budget-W, got g={g}, W={W}, "
                f"k={k}, budget={_CANDIDATE_BUDGET}."
            )
        c = _CANDIDATE_BUDGET - W

        device = q_li.device

        # ---- 1. mean proxy (dequant -> segment mean -> requant) ----------
        q_dq = q_li.view(N_in, H, D)[:N].to(torch.bfloat16) * q_li_scale.view(
            N_in, H, 1
        )[:N].to(torch.bfloat16)  # [N, H, D], Hadamard domain
        q_bar, w_bar = _segment_mean(q_dq, weights[:N], group_start, counts)

        # Requant the proxy exactly like the native q side (sfa_v1.py
        # npu_dynamic_quant on [tokens*heads, D]).
        q_bar_flat = q_bar.reshape(-1, D)
        q_bar_q, q_bar_scale = torch_npu.npu_dynamic_quant(
            q_bar_flat, dst_type=sfa_impl.c8_k_cache_dtype
        )
        q_bar_scale = q_bar_scale.to(sfa_impl.c8_k_scale_cache_dtype)

        # ---- 2. proxy scan (native op, R proxies, key len = t0) ----------
        packed_kv_cache = getattr(sfa_impl, "enable_sparse_sfa_c8", False)
        indexer_cache_idx = 1 if packed_kv_cache else 2
        indexer_scale_cache_idx = 2 if packed_kv_cache else 3
        topk_candidates = torch_npu.npu_quant_lightning_indexer(
            query=q_bar_q.view(R, H, D),
            key=kv_cache[indexer_cache_idx],
            weights=w_bar,
            query_dequant_scale=q_bar_scale.view(R, H),
            key_dequant_scale=kv_cache[indexer_scale_cache_idx].squeeze(2),
            actual_seq_lengths_query=torch.arange(1, R + 1, dtype=torch.int32, device=device),
            actual_seq_lengths_key=proxy_key_lens.to(torch.int32),
            block_table=attn_metadata.block_table,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=c,
            sparse_mode=3,
        )
        C = topk_candidates[:, 0, :]  # [R, c], 0-based, causally clean

        # ---- 3. refine: joint C∪W_t scoring + joint top-k ---------------
        topk_indices = _refine_topk(
            q_dq,
            weights[:N],
            C,
            window_pos,
            positions_q,
            req_ids,
            kv_cache,
            indexer_cache_idx,
            indexer_scale_cache_idx,
            attn_metadata.block_table,
            attn_metadata.block_size,
            k,
            W,
            N,
        )

        # Graph padding guard (shape-based branch, capture-safe: N_in/N are
        # Python ints, and capture-time dummy batches have N_in == N):
        # padded rows [N, N_in) get -1 tails so the output row count matches
        # the native path (num_input_tokens).
        if N_in > N:
            row_pad = torch.full(
                (N_in - N, 1, topk_indices.shape[-1]),
                -1,
                dtype=topk_indices.dtype,
                device=device,
            )
            topk_indices = torch.cat([topk_indices, row_pad], dim=0)

        # use_index_cache width guard: the read side returns the full buffer
        # width, so pad the output to the buffer width with -1 tails
        # (design doc §4.4, countermeasure a).
        if sfa_impl.use_index_cache and sfa_impl.topk_indices_buffer is not None:
            buf_width = sfa_impl.topk_indices_buffer.shape[-1]
            if topk_indices.shape[-1] < buf_width:
                pad = torch.full(
                    (topk_indices.shape[0], 1, buf_width - topk_indices.shape[-1]),
                    -1,
                    dtype=topk_indices.dtype,
                    device=device,
                )
                topk_indices = torch.cat([topk_indices, pad], dim=-1)

        logger.debug(
            "PIVOT refine: N=%d R=%d g=%d W=%d c=%d k=%d", N, R, g, W, c, k
        )
        return topk_indices


def _segment_mean(q_dq: torch.Tensor, weights: torch.Tensor, group_start, counts):
    """Segment (per-request) mean via cumsum + boundary diff (graph-safe)."""
    R = counts.shape[0]
    H, D = q_dq.shape[1], q_dq.shape[2]
    HD = H * D
    q_cum = torch.cat(
        [torch.zeros(1, HD, dtype=q_dq.dtype, device=q_dq.device),
         torch.cumsum(q_dq.view(-1, HD), dim=0)]
    )  # [N+1, HD]
    ends = (group_start + counts).to(torch.long)  # [R] == cum_query_lens
    starts = group_start.to(torch.long)  # [R]
    q_sum = q_cum[ends] - q_cum[starts]  # [R, HD]
    counts_f = counts.to(q_dq.dtype).clamp(min=1).unsqueeze(-1)  # [R, 1]
    q_bar = (q_sum / counts_f).view(R, H, D)

    w_cum = torch.cat(
        [torch.zeros(1, weights.shape[1], dtype=weights.dtype, device=weights.device),
         torch.cumsum(weights, dim=0)]
    )
    w_bar = (w_cum[ends] - w_cum[starts]) / counts_f  # [R, H]
    return q_bar, w_bar


def _refine_topk(
    q_dq, weights, C, window_pos, positions_q, req_ids,
    kv_cache, indexer_cache_idx, indexer_scale_cache_idx,
    block_table, block_size, k, W, N,
):
    """Joint C∪W_t scoring with the operator's formula, then joint top-k."""
    device = q_dq.device
    R, c = C.shape

    k_cache = kv_cache[indexer_cache_idx]  # [B, S, 1, D] FP8 (N2 = 1)
    kc = k_cache.view(-1, k_cache.shape[-1])  # [B*S, D]
    k_scale_flat = kv_cache[indexer_scale_cache_idx].squeeze(2).reshape(-1)  # [B*S]

    r = torch.arange(R, dtype=torch.int64, device=device)
    c_safe = C.clamp(min=0).to(torch.int64)  # -1 slots (short prefix) clamped for gather
    slots_c = (
        block_table[r.unsqueeze(1), c_safe // block_size] * block_size + c_safe % block_size
    )  # [R, c]
    k_cand = kc[slots_c.reshape(-1)].view(R, c, -1)  # [R, c, D] FP8
    k_scale = k_scale_flat[slots_c.reshape(-1)].view(R, c)
    k_dq = k_cand.to(torch.bfloat16) * k_scale.to(torch.bfloat16).unsqueeze(-1)  # [R, c, D]

    # C-side scores: [N, c], chunked over R (_REFINE_CHUNK).
    score_c = _score_c_side(q_dq, weights, k_dq, req_ids, c)

    # W-side scores: [N, W] small per-query window bmm.
    w_safe = window_pos.clamp(min=0).to(torch.int64)
    slots_w = (
        block_table[req_ids.unsqueeze(1), w_safe // block_size] * block_size + w_safe % block_size
    )  # [N, W]
    k_win = kc[slots_w.reshape(-1)].view(N, W, -1)
    s_win = k_scale_flat[slots_w.reshape(-1)].view(N, W)
    k_win_dq = k_win.to(torch.bfloat16) * s_win.to(torch.bfloat16).unsqueeze(-1)
    att_w = torch.relu(
        torch.matmul(q_dq, k_win_dq.transpose(1, 2))
    )  # [N, H, W]
    score_w = (att_w * weights.unsqueeze(-1)).sum(dim=1)  # [N, W]

    # Invalid-slot masks at score level (pre-topk, zero slot waste):
    #  - C side: short-prefix -1, window duplicates (C∪W union dedup),
    #    future keys (> positions_q; source removal makes this a no-op
    #    guard for counts/seq_lens edge cases)
    #  - W side: sequence-start out-of-range columns
    C_n = C[req_ids]  # [N, c]
    in_win = (C_n.unsqueeze(1) == window_pos.unsqueeze(-1)).any(dim=1)  # [N, c]
    invalid_c = (C_n < 0) | in_win | (C_n > positions_q.unsqueeze(-1))
    score_c = score_c.masked_fill(invalid_c, float("-inf"))
    score_w = score_w.masked_fill(window_pos < 0, float("-inf"))

    # Joint top-k over the deduped union (window columns carry their own
    # true positions), then -1 sanitation for deficient valid candidates.
    score_all = torch.cat([score_w, score_c], dim=-1)  # [N, W + c]
    pos_all = torch.cat([window_pos, C_n], dim=-1)  # [N, W + c]
    vals, cols = torch.topk(score_all, k, dim=-1)
    true_pos = pos_all.gather(1, cols)
    true_pos = true_pos.masked_fill(vals == float("-inf"), -1)

    return true_pos.view(N, 1, k).to(torch.int32)


def _score_c_side(q_dq, weights, k_dq, req_ids, c):
    """score_c[n, j] = sum_h w[n, h] * ReLU(q[n, h] . k_dq[req(n), j]).

    Fast path assumes a uniform group size g (steady-state MTP and graph
    decode): queries of one request are reshaped into one bmm batch,
    chunked over requests. For eager batches with non-uniform draft
    counts we fall back to a per-request loop (correct, host-driven,
    never graph-captured).
    """
    N, H, D = q_dq.shape
    R = k_dq.shape[0]
    capturing = False
    try:
        capturing = torch.npu.is_current_stream_capturing()
    except Exception:
        pass
    if capturing:
        # Graph decode guarantees uniform groups (patch_cudagraph asserts
        # divisibility); skip the host sync below during capture.
        uniform = True
    else:
        counts = torch.bincount(req_ids, minlength=R)
        uniform = bool((counts == counts[0]).all().item())
    if uniform:
        g = N // R
        out = torch.empty(N, c, dtype=q_dq.dtype, device=q_dq.device)
        for s in range(0, R, _REFINE_CHUNK):
            e = min(s + _REFINE_CHUNK, R)
            rows = (e - s) * g
            score = torch.matmul(
                q_dq[s * g:e * g].reshape(e - s, g * H, D), k_dq[s:e].transpose(1, 2)
            )  # [chunk, g*H, c]
            att = torch.relu(score).view(e - s, g, H, c)
            score_c = (att * weights[s * g:e * g].view(e - s, g, H, 1)).sum(dim=2)
            out[s * g:e * g] = score_c.reshape(rows, c)
        return out

    # Non-uniform fallback: per-request matmul (eager only).
    logger.warning_once(
        "PIVOT: non-uniform query groups; falling back to per-request scoring."
    )
    counts = torch.bincount(req_ids, minlength=R)
    kT = k_dq.transpose(1, 2)  # [R, D, c]
    out = torch.empty(N, c, dtype=q_dq.dtype, device=q_dq.device)
    start = 0
    for r in range(R):
        end = start + int(counts[r].item())
        att = torch.relu(torch.matmul(q_dq[start:end], kT[r]))  # [rows, H, c]
        out[start:end] = (att * weights[start:end].unsqueeze(-1)).sum(dim=1)
        start = end
    return out
