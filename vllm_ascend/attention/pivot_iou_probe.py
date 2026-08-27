# SPDX-License-Identifier: Apache-2.0
"""PIVOT step-to-step IoU capture probe (measurement only, off by default).

Purpose
-------
The engineering premise for the "temporal-persistent candidate pool"
innovation (survey innovation #1) is that the coarse/refine top-k sets
overlap heavily between adjacent decode steps. The literature (GVR
2604.22312, FlexiCache 2511.00868) confirms this on GPU with a
head-dependent nuance, but it must be measured on GLM-5.2-w4a8 BF16
non-C8 (A3 / 910B) through this repo's PIVOT path before it is treated as
an engineering fact. This probe captures the exact sets PIVOT consumes:

  * coarse set C         [K, 4096]   per-request (mean-proxy screen)
  * refine set topk_dec  [D, 1, 2048] per-query-row (torch refine)

Key design decisions
--------------------
* Capture is keyed by (layer_name, step) -- the indexer runs once per SFA
  layer per decode step, and step-to-step overlap is only meaningful within
  a layer (the literature's per-layer finding). layer_name tags which SFA
  impl instance produced the set.
* Cross-step alignment of the same request is NOT done here: the probe
  records per-request identity anchors (cumulative query start = the
  request's row offset, seq_len) and the offline analyzer chains them.
  cum_start increases monotonically for a live request and resets to 0 for
  a new request, so chains are unambiguous.
* Captured data is buffered on the Python side and flushed to .npz every
  _FLUSH_STEPS steps so the hot path never grows unbounded. Two .cpu()
  copies per layer per step is the only runtime cost, and only when the
  probe is enabled.
* Graph capture is skipped entirely (torch.npu.is_current_stream_capturing
  -- same guard as pivot_indexer._capturing): capturing inside an NPUGraph
  would freeze the snapshot to a single step.
"""

from __future__ import annotations

import os

import numpy as np
import torch

# Number of steps (per layer) to buffer before flushing a .npz file.
_FLUSH_STEPS = 128


def _enabled() -> bool:
    return bool(int(os.getenv("VLLM_ASCEND_PIVOT_IOU_DUMP", "0")))


def _out_dir() -> str:
    return os.getenv("VLLM_ASCEND_PIVOT_IOU_DIR", "/tmp/pivot_iou/")


def _capturing() -> bool:
    try:
        return torch.npu.is_current_stream_capturing()
    except Exception:
        return False


class _IoUProbe:
    """Buffers per-(layer, step) PIVOT top-k sets and flushes to .npz."""

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._steps: dict[str, int] = {}  # layer_name -> flushed step count
        self._flushes = 0
        self._out = _out_dir()
        os.makedirs(self._out, exist_ok=True)

    def capture(
        self,
        layer_name: str,
        seq_lens: torch.Tensor,     # [K] key-side cumulative lengths
        cum: torch.Tensor,          # [K] query-side cumulative ends
        coarse: torch.Tensor,       # [K, 4096] 0-based positions, -1 padded
        refine: torch.Tensor,       # [D, 1, 2048] 0-based positions, -1 padded
        K: int,
        D: int,
        g: int,
    ) -> None:
        # decode segment rows [0, K) / [0, D) only; prefill tail excluded.
        step = self._steps.get(layer_name, 0)
        self._steps[layer_name] = step + 1
        # Query start offset of request r = cumulative end of request r-1
        # (request 0 starts at 0). Monotone across steps for a live request:
        # each decode step appends g query rows, so the start shifts +g.
        cum_k = cum[:K].cpu().numpy().astype(np.int64)
        cum_start = np.concatenate([np.zeros(1, dtype=np.int64), cum_k[:-1]])
        self._rows.append({
            "layer": layer_name,
            "step": step,
            "g": g,
            "seq_lens": seq_lens[:K].cpu().numpy().astype(np.int64),
            "cum_start": cum_start,
            "coarse": coarse[:K].cpu().numpy().astype(np.int64),
            "refine": refine[:D, 0].cpu().numpy().astype(np.int64),
        })
        if step and step % _FLUSH_STEPS == 0:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        rows = self._rows
        self._rows = []
        path = os.path.join(self._out, f"pivot_iou_{self._flushes:05d}.npz")
        np.savez_compressed(path, rows=rows)
        self._flushes += 1

    def finish(self) -> None:
        self._flush()


# Module-level singleton; import is cheap (np/torch already loaded by
# pivot_indexer callers), capture() no-ops when the env switch is off.
_probe: _IoUProbe | None = None


def capture(
    layer_name: str,
    seq_lens: torch.Tensor,
    cum: torch.Tensor,
    coarse: torch.Tensor,
    refine: torch.Tensor,
    K: int,
    D: int,
    g: int,
) -> None:
    """Entry point called from PivotIndexer.select_topk before returning.

    No-ops unless VLLM_ASCEND_PIVOT_IOU_DUMP=1 and not in graph capture.
    """
    global _probe
    if not _enabled():
        return
    if _capturing():
        return
    if _probe is None:
        _probe = _IoUProbe()
    _probe.capture(layer_name, seq_lens, cum, coarse, refine, K, D, g)


def finish() -> None:
    global _probe
    if _probe is not None:
        _probe.finish()
        _probe = None
