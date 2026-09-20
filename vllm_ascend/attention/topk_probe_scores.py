# -*- coding: utf-8 -*-
"""Score-bounds probe: dump per-(block, head) dot-product extremes + the full
per-key score vector for ONE query row per sampled step.

Purpose (idea_mcts/report.html 第 0 批 / 跑 1+跑 2): the existing topk probe
dumps indices only, which cannot validate any ceiling-bound variant (跳块法/
提前收工/合并候选池/复核检查) offline -- those need the actual per-key scores.
This module dumps, for a sampled query row:

  * maxdot [NB, H]  -- max_j <q_h, k_j> within each block, per head
  * mindot [NB, H]  -- min_j <q_h, k_j> within each block, per head
  * score  [L]      -- the exact native score vector sum_h w_h*ReLU(<q_h,k_j>)
  * q_row/w_row     -- the query rows and weights themselves (small)

From (maxdot, mindot, w) every ceiling variant is recomputable offline:

  B(b) = sum_{w>0} w_h*max(0, maxdot_hb) + sum_{w<0} |w_h|*max(0, -mindot_hb)

(ReLU monotonicity lets max/min penetrate the nonlinearity; the per-head
max/min form subsumes the mean+radius and sign-aware variants).

Design constraints mirror baseline_topk_probe.py: env-gated, eager-only
(skips graph capture), sampled steps only, never raises, one tracked request
per sampled step (the first tracked one -- gate decisions are per-layer
statistics, one request is enough).

Usage: call ``capture_score_bounds`` from the two indexer call sites
(pivot path: ``PivotIndexer.select_topk`` after q_dq/weights are built;
native path: the device_op fallback branch after q/weights are built).
Both call sites are gated here by ``VLLM_ASCEND_SCORE_PROBE`` (default off).
"""

from __future__ import annotations

import os

import numpy as np
import torch

_SCORE_BLOCK = 128          # must match the indexer cache block size
_DUMP_DIR = "/tmp/score_probe"
_ENV_ON = "VLLM_ASCEND_SCORE_PROBE"
_ENV_STRIDE = "VLLM_ASCEND_SCORE_PROBE_STRIDE"      # sample every Nth step
_ENV_MAX_DUMP = "VLLM_ASCEND_SCORE_PROBE_MAX"       # stop after this many dumps

_state = {"counter": 0, "dumped": 0, "warned": False}


def _enabled() -> bool:
    return os.getenv(_ENV_ON, "0") == "1"


def _probe_device() -> torch.device:
    return torch.device("npu" if torch.npu.is_available() else "cuda")


def _dump_path(step: int, layer_name: str) -> str:
    safe = layer_name.replace("/", "_").replace(".", "_")
    rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
    return os.path.join(_DUMP_DIR, f"score__{safe}__step{step}__r{rank}.npz")


def capture_score_bounds(layer_name: str, step: int, q_row: torch.Tensor,
                         w_row: torch.Tensor, k_cache: torch.Tensor,
                         block_table_row: torch.Tensor, seq_len: int) -> None:
    """Dump dot extremes + the score vector for one query row. Never raises.

    q_row   [H, Dh] bf16 -- one query row, per-head (pre-ReLU dot form)
    w_row   [H]          -- the per-head weights of the same row
    k_cache [B, S, 1, Dh]-the indexer key cache (paged, C8 views already
            dequantized upstream -- the dot here must use the SAME
            dequantization as the fused op, see report 铁律②)
    block_table_row [nb] -- physical block ids of this request, logical order
    seq_len int          -- this request's valid key count (prompt+generated)
    """
    if not _enabled():
        return
    try:
        _state["counter"] += 1
        stride = int(os.getenv(_ENV_STRIDE, "8"))
        if _state["counter"] % max(1, stride) != 0:
            return
        max_dumps = int(os.getenv(_ENV_MAX_DUMP, "64"))
        if _state["dumped"] >= max_dumps:
            return
        if torch.npu.is_current_stream_capturing():
            return  # graph capture is unsafe for D2H + I/O; eager runs only

        dev = _probe_device()
        H = q_row.shape[0]
        q = q_row.detach().to(torch.float32)
        w = w_row.detach().to(torch.float32)
        bt = block_table_row.detach().to(torch.int64)

        # gather this request's keys in logical order: [nb*S, Dh] -> [:seq_len]
        k_pages = k_cache.detach().to(torch.float32).view(k_cache.shape[0], -1,
                                                          k_cache.shape[-1])
        k_flat = k_pages[bt].reshape(-1, k_pages.shape[-1])[:seq_len]
        L = k_flat.shape[0]
        if L <= 0:
            return

        dots = torch.einsum("hd,jd->hj", q, k_flat)             # [H, L]
        relu = torch.clamp(dots, min=0.0)
        score = (w[:, None] * relu).sum(dim=0)                  # [L]

        nb = (L + _SCORE_BLOCK - 1) // _SCORE_BLOCK
        pad = nb * _SCORE_BLOCK - L
        if pad:
            dots = torch.cat([dots, dots.new_full((H, pad), float("-inf"))], dim=1)
        d3 = dots.view(H, nb, _SCORE_BLOCK)
        maxdot = d3.max(dim=2).values                           # [H, NB]
        mindot = d3.min(dim=2).values                           # [H, NB]

        os.makedirs(_DUMP_DIR, exist_ok=True)
        np.savez_compressed(
            _dump_path(step, layer_name),
            layer_name=layer_name, step=step, seq_len=L, top_k=K_DEFAULT(),
            q_row=q.cpu().numpy(), w_row=w.cpu().numpy(),
            maxdot=maxdot.cpu().numpy().T.astype(np.float16),   # [NB, H]
            mindot=mindot.cpu().numpy().T.astype(np.float16),   # [NB, H]
            score=score.cpu().numpy().astype(np.float16),       # [L]
            block_table_row=bt.cpu().numpy(),
            block_size=_SCORE_BLOCK,
        )
        _state["dumped"] += 1
    except Exception:  # noqa: BLE001 -- diagnostic path must never break serving
        if not _state["warned"]:
            _state["warned"] = True
            try:
                from vllm.logger import logger
                logger.warning("score probe: capture failed once; disabled "
                               "for the rest of the run")
            except Exception:
                pass


def K_DEFAULT() -> int:
    """Native top-k (2048); the analyzer reads it from the dump payload."""
    return 2048
