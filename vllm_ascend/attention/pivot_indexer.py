# SPDX-License-Identifier: Apache-2.0
"""PIVOT-Refine indexer: mean-proxy coarse screen (top-4096 + local-window,
via npu_indexer_coarse_screen) + per-query refine (top-2048, via
npu_indexer_refine; op emits KV positions directly).

Batch is reordered decode-first; select_topk returns the decode head's topk,
the caller merges the prefill tail (select_topk_prefill / native indexer).
Decode path is host-geometry only (zero D2H) -> FULL_DECODE_ONLY
capturable. Lossless: L+g <= 2048 reproduces the full prefix; k fixed at
2048 (native sparse_count). BF16 only (C8 falls back to native).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

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

_MAX_GROUP = 16

def _request_counts(cum: torch.Tensor) -> torch.Tensor:
    """Per-request query-token counts from cumulative ends (query_start_loc[1:])."""
    starts = torch.cat([torch.zeros(1, dtype=cum.dtype, device=cum.device), cum[:-1]])
    return (cum - starts).to(torch.int64)


def build_prefill_geometry(cum, seq_lens, block_table, D):
    """Per-STEP prefill-PIVOT geometry (computed once in metadata build,
    consumed by every layer's select_topk_prefill): positional groups of g
    over the prefill tail [D, N). Returns None when there is no tail."""
    g = envs.VLLM_ASCEND_PIVOT_PREFILL_GROUP
    pre_req = cum > D
    R_pre = int(pre_req.sum())
    if R_pre == 0:
        return None
    device = cum.device
    tail_cum = cum[pre_req] - D
    tail_q = _request_counts(tail_cum)
    n_r = (tail_q + g - 1) // g
    P = int(n_r.sum())
    group_sizes = torch.full((P,), g, dtype=torch.int64, device=device)
    group_sizes[n_r.cumsum(0) - 1] = tail_q - (n_r - 1) * g
    req_base = n_r.cumsum(0) - n_r
    L_r = seq_lens[pre_req] - tail_q
    group_j = torch.arange(P, device=device) - torch.repeat_interleave(req_base, n_r)
    return SimpleNamespace(
        g=g,
        N_tail=int(tail_cum[-1]),
        R_pre=R_pre,
        P=P,
        group_sizes=group_sizes,
        aslq_refine=group_sizes.cumsum(0),
        group_ids=torch.repeat_interleave(torch.arange(P, device=device), group_sizes),
        group_start=torch.repeat_interleave(L_r, n_r) + group_j * g,
        group_bt=block_table[pre_req].repeat_interleave(n_r, dim=0),
    )


class PivotIndexer:
    """PIVOT-Refine top-k selection (op-reuse core, BF16)."""

    @staticmethod
    def select_topk(
        sfa_impl,
        q_li: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: tuple,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> torch.Tensor | None:
        """Decode-head topk_indices [D, 1, 2048]; None -> caller falls back
        to the native indexer. Geometry comes from metadata (host-only, zero
        D2H) -> FULL_DECODE_ONLY capturable."""
        if getattr(sfa_impl, "enable_sparse_li_c8", False):
            logger.warning_once(
                "PIVOT: enable_sparse_li_c8 is set; falling back to the "
                "native indexer (PIVOT supports the BF16 path only)."
            )
            return None

        cum = actual_seq_lengths_query  # [R], cumulative ends
        seq_lens = actual_seq_lengths_key  # [R], == L + g at indexer time
        if cum.shape[0] == 0:
            return None

        K = attn_metadata.num_decodes
        D = attn_metadata.num_decode_tokens
        g = attn_metadata.pivot_decode_g
        if K == 0 or g is None or K * g != D or not 2 <= g <= _MAX_GROUP:
            logger.warning_once(
                "PIVOT: decode head geometry K=%d D=%d g=%s is not a "
                "uniform grouped-decode segment; bailing the whole batch to "
                "the native indexer.", K, D, g)
            return None

        q_dq = q_li[:D]  # raw BF16 [D, H, Dh]
        H, Dh = q_dq.shape[1], q_dq.shape[2]

        # Coarse: group mean proxy -> top-4096 over [0, seq_lens-g) + the
        # local-window union injection (window = [aslk-(g-1), aslk+own)).
        q_bar = q_dq.view(K, g, H, Dh).mean(dim=1)
        w_bar = weights[:D].view(K, g, H).mean(dim=1)
        C, aslk_op = torch.ops._C_ascend.npu_indexer_coarse_screen(
            q_bar,
            w_bar,
            kv_cache[2],
            actual_seq_lengths_query=cum[:K].to(torch.int32),
            # Graph replay pads seq_lens rows [num_reqs, K) with 0
            # (model_runner_v1.py:1264); subtracting g turns pad rows into a
            # NEGATIVE S2 walk bound and the key gather runs out of the KV
            # pool -- MTE invalid-GM AICORE fault on the first sub-capacity
            # replay (2026-09-21 19:41, fault kernel IndexerCoarseScreen).
            # The floor is 1, NOT 0: an aslk=0 (empty-walk) row corrupts its
            # NEIGHBORS' candidate sets in the coarse kernel (graph_smoke_
            # service_shape.py S6b 2026-09-21: pad=0 -> real-row C_diff=2507
            # while the eager-determinism control is clean; pad=1/4/4096 all
            # clean) -- clamp-to-0 traded the crash for silent candidate
            # corruption. Real rows have aslk = L >= 1, so the floor only
            # touches pads; a pad row walks 1 stale-but-pooled block id --
            # memory-safe, and its output is masked downstream.
            actual_seq_lengths_key=torch.clamp(seq_lens[:K] - g, min=1).to(torch.int32),
            block_table=attn_metadata.block_table[:K],
            coarse_count=_COARSE_BUDGET,
            has_window=1,
            group_size=g,
        )

        # Refine: per-query scores over C, top-2048. The op emits KV
        # POSITIONS directly (2026-09-20 kernel change) — no caller-side
        # column gather anymore.
        topk_dec = torch.ops._C_ascend.npu_indexer_refine(
            q_dq,
            kv_cache[2],
            weights[:D],
            C,
            actual_seq_lengths_query=cum[:K].to(torch.int32),
            actual_seq_lengths_key=aslk_op.to(torch.int32),
            block_table=attn_metadata.block_table[:K],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=_REFINE_BUDGET,
        )
        return topk_dec


    @staticmethod
    def select_topk_prefill(
        sfa_impl,
        q_li: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: tuple,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        prefill_tail_start: int,
    ) -> torch.Tensor | None:
        """Prefill-tail topk_indices via prefill-PIVOT; None -> native
        fallback. Consumes the per-step geometry from
        metadata.pivot_prefill (build_prefill_geometry); prefill is always
        eager."""
        if getattr(sfa_impl, "enable_sparse_li_c8", False):
            logger.warning_once(
                "PIVOT prefill: enable_sparse_li_c8 is set; falling back to "
                "the native indexer (PIVOT supports the BF16 path only).")
            return None

        geo = attn_metadata.pivot_prefill
        if geo is None or not 2 <= geo.g <= _MAX_GROUP:
            return None
        D = prefill_tail_start

        # Per-layer mean proxy (q/weights differ per layer, so this cannot be
        # hoisted to metadata): index_add_ takes a 1-D index -- no shape
        # expansion, unlike scatter_add_ (whose expanded index OOMed on large
        # tails). Numerically the same accumulation as the validated
        # scatter_add path.
        q_dq = q_li[D:]
        w_t = weights[D:]
        H, Dh = q_dq.shape[1], q_dq.shape[2]
        q_bar = q_dq.new_zeros(geo.P, H, Dh) \
            .index_add_(0, geo.group_ids, q_dq) / geo.group_sizes.view(-1, 1, 1)
        w_bar = w_t.new_zeros(geo.P, H) \
            .index_add_(0, geo.group_ids, w_t) / geo.group_sizes.view(-1, 1)

        C, aslk_op = torch.ops._C_ascend.npu_indexer_coarse_screen(
            q_bar,
            w_bar,
            kv_cache[2],
            actual_seq_lengths_query=geo.aslq_refine.to(torch.int32),
            actual_seq_lengths_key=geo.group_start.to(torch.int32),
            block_table=geo.group_bt,
            coarse_count=_COARSE_BUDGET,
            has_window=1,
            group_size=geo.g,
        )
        topk_pre = torch.ops._C_ascend.npu_indexer_refine(
            q_dq, kv_cache[2], w_t, C,
            actual_seq_lengths_query=geo.aslq_refine.to(torch.int32),
            actual_seq_lengths_key=aslk_op.to(torch.int32),
            block_table=geo.group_bt,
            layout_query="TND", layout_key="PA_BSND",
            sparse_count=_REFINE_BUDGET,
        )
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

    Applied by the caller on the combined decode+prefill result. Two pads,
    both -1 tails:
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

