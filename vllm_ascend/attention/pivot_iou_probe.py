# SPDX-License-Identifier: Apache-2.0
"""PIVOT step-to-step top-k change probe (measurement only, off by default).

What it measures
----------------
The engineering premise for the "temporal-persistent candidate pool"
innovation is that adjacent decode steps pick almost the same coarse/refine
top-k sets. We measure exactly that, WITHOUT storing the big arrays: for each
SFA layer, for each adjacent pair of decode steps, we count how many top-k
indices changed (the symmetric-difference size |A Delta B|) for the same
request, then throw the arrays away. Only these small integer counts are
recorded.

Design constraints (hard)
-------------------------
1. NON-BLOCKING: capture never blocks the decode hot path. Serialization +
   disk I/O run on a daemon writer thread; capture only puts a tiny record
   onto a queue and returns.
2. BOUNDED MEMORY: we never accumulate the [K,4096]/[D,2048] arrays across
   steps. We keep exactly ONE previous snapshot per layer (to compare against
   the current step), compute the diff counts, then keep only the counts.
   Peak memory = per-layer snapshot + small queue of integer records.
3. SIMPLE: no fancy alignment/grouping/dist-decay logic here -- just per-layer
   adjacent-step change counts. Keep it dumb and reliable.
4. Graph capture is skipped (torch.npu.is_current_stream_capturing -- same
   guard as pivot_indexer._capturing).
"""

from __future__ import annotations

import atexit
import os
import queue
import threading

import numpy as np
import torch

# Number of diff records to buffer on the writer before writing a .npz.
_FLUSH_STEPS = 256


def _enabled() -> bool:
    return bool(int(os.getenv("VLLM_ASCEND_PIVOT_IOU_DUMP", "0")))


def _out_dir() -> str:
    return os.getenv("VLLM_ASCEND_PIVOT_IOU_DIR", "/tmp/pivot_iou/")


def _capturing() -> bool:
    try:
        return torch.npu.is_current_stream_capturing()
    except Exception:
        return False


def _diff_count(a: np.ndarray, b: np.ndarray) -> int:
    """|A Delta B| -- number of top-k positions that differ (ignores -1 pad)."""
    sa = set(int(x) for x in a if x >= 0)
    sb = set(int(x) for x in b if x >= 0)
    return len(sa ^ sb)


class _IoUProbe:
    """Per-layer adjacent-step top-k change counter. Non-blocking, bounded mem."""

    _STOP = object()

    def __init__(self) -> None:
        # One previous snapshot per layer: {g, cum_start, coarse, refine}
        self._prev: dict[str, dict] = {}
        self._steps: dict[str, int] = {}
        self._out = _out_dir()
        self._q: "queue.Queue" = queue.Queue()
        self._worker = threading.Thread(target=self._writer, daemon=True)
        self._worker.start()
        try:
            os.makedirs(self._out, exist_ok=True)
        except OSError as e:
            print(f"PIVOT[iou] cannot create {self._out}: {e!r}")

    def _writer(self) -> None:
        buf: list[dict] = []
        flushes = 0
        while True:
            item = self._q.get()
            if item is self._STOP:
                self._dump(buf, flushes)
                return
            buf.append(item)
            if len(buf) >= _FLUSH_STEPS:
                flushes = self._dump(buf, flushes)
                buf = []

    def _dump(self, rows: list[dict], flushes: int) -> int:
        if not rows:
            return flushes
        path = os.path.join(self._out, f"pivot_iou_{flushes:05d}.npz")
        try:
            np.savez_compressed(path, rows=rows)
        except Exception as e:  # never let a dump failure kill the writer
            print(f"PIVOT[iou] dump failed: {e!r}")
            return flushes
        return flushes + 1

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

        cum_k = cum[:K].cpu().numpy()
        cum_start = np.concatenate([np.zeros(1, dtype=cum_k.dtype), cum_k[:-1]])
        coarse_c = coarse[:K].cpu().numpy()
        refine_c = refine[:D, 0].cpu().numpy()

        prev = self._prev.get(layer_name)
        if prev is not None and prev["g"] == g:
            # Same request at step t+1 has cum_start == s + g.
            a_idx = {int(s): i for i, s in enumerate(prev["cum_start"])}
            b_idx = {int(s): i for i, s in enumerate(cum_start)}
            cdiff = rdiff = matched = 0
            for s, ia in a_idx.items():
                ib = b_idx.get(s + g)
                if ib is None:
                    continue
                matched += 1
                cdiff += _diff_count(prev["coarse"][ia], coarse_c[ib])
                ra = prev["refine"][ia * g:(ia + 1) * g]
                rb = refine_c[ib * g:(ib + 1) * g]
                if ra.shape[0] == g and rb.shape[0] == g:
                    rdiff += sum(_diff_count(ra[i], rb[i]) for i in range(g))
            if matched:
                self._q.put({
                    "layer": layer_name,
                    "step": step,
                    "g": g,
                    "coarse_diff": cdiff,
                    "refine_diff": rdiff,
                    "n_req": matched,
                })

        # Keep this step's snapshot as the next comparison baseline.
        self._prev[layer_name] = {
            "g": g,
            "cum_start": cum_start,
            "coarse": np.ascontiguousarray(coarse_c, dtype=np.int64),
            "refine": np.ascontiguousarray(refine_c, dtype=np.int64),
        }

    def finish(self) -> None:
        self._q.put(self._STOP)
        self._worker.join(timeout=30)


# Module-level singleton; import is cheap, capture() no-ops when env is off.
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
    """Entry point called from PivotIndexer.select_topk before returning."""
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


# Flush the last (partial) batch on normal process exit. Idempotent; no-op
# when the probe was never enabled (env off / graph capture skipped).
atexit.register(finish)
