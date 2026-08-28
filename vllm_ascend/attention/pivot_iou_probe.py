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
3. EXACT MATCHING: adjacent steps are paired by the REAL request ID string
   (threaded in from the model runner's input_batch.req_ids each step, see
   set_step_request_ids). Request IDs are globally unique, so no heuristic
   (query start / key length / batch position) is used -- under continuous
   batching those all collide and would mispair requests. If no request IDs
   are available for a step, capture is skipped rather than guessed.
4. SIMPLE: no fancy alignment/grouping/dist-decay logic here -- just per-layer
   adjacent-step change counts for the same real request. Keep it dumb.
5. Graph capture is skipped (torch.npu.is_current_stream_capturing -- same
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
        seq_lens: torch.Tensor,     # [R] key-side cumulative lengths (unused here)
        cum: torch.Tensor,          # [R] query-side cumulative ends (unused here)
        coarse: torch.Tensor,       # [K, 4096] 0-based positions, -1 padded
        refine: torch.Tensor,       # [D, 1, 2048] 0-based positions, -1 padded
        K: int,
        D: int,
        g: int,
    ) -> None:
        # decode segment rows [0, K) / [0, D) only; prefill tail excluded.
        step = self._steps.get(layer_name, 0)
        self._steps[layer_name] = step + 1

        # Real request IDs for THIS step, in decode-segment row order (threaded
        # in from the model runner each step via set_step_request_ids). Exact
        # string matching only -- no heuristic. Skip if unavailable.
        cur_ids = _current_req_ids(K)
        if cur_ids is None:
            return

        coarse_c = coarse[:K].cpu().numpy()
        refine_c = refine[:D, 0].cpu().numpy()

        prev = self._prev.get(layer_name)
        cdiff = rdiff = matched = K_prev = 0
        if prev is not None and prev["g"] == g:
            prev_ids = prev["req_ids"]        # [P] this layer's prev request ids
            K_prev = len(prev_ids)
            cur_idx = {rid: c for c, rid in enumerate(cur_ids)}  # unique by id
            for p, rid in enumerate(prev_ids):
                c = cur_idx.get(rid)
                if c is None:
                    continue  # request finished / not in this step -> no pair
                matched += 1
                cdiff += _diff_count(prev["coarse"][p], coarse_c[c])
                ra = prev["refine"][p * g:(p + 1) * g]
                rb = refine_c[c * g:(c + 1) * g]
                if ra.shape[0] == g and rb.shape[0] == g:
                    rdiff += sum(_diff_count(ra[i], rb[i]) for i in range(g))
            if matched:
                self._q.put({
                    "layer": layer_name,
                    "step": step,
                    "g": g,
                    "K": K_prev,          # prev requests available to match
                    "n_req": matched,     # how many had the same real request id
                    "coarse_diff": cdiff,
                    "refine_diff": rdiff,
                })

        # Keep this step's snapshot as the next comparison baseline.
        self._prev[layer_name] = {
            "g": g,
            "req_ids": list(cur_ids),
            "coarse": np.ascontiguousarray(coarse_c, dtype=np.int64),
            "refine": np.ascontiguousarray(refine_c, dtype=np.int64),
        }

    def finish(self) -> None:
        self._q.put(self._STOP)
        self._worker.join(timeout=30)


# Module-level singleton; import is cheap, capture() no-ops when env is off.
_probe: _IoUProbe | None = None

# Request IDs of the CURRENT decode step, in batch (row) order, threaded in by
# the model runner once per step (see set_step_request_ids). None until the
# runner has fed a step -- capture() skips (never guesses) until then.
_CUR_REQ_IDS: list[str] | None = None


def set_step_request_ids(req_ids) -> None:
    """Feed this step's request IDs (batch order) from the model runner.

    Called once per execute_model, before the model forward. The probe (fired
    per layer inside the forward) then matches adjacent steps by exact request
    ID. No-op cost when the probe env is off (just a list copy).
    """
    global _CUR_REQ_IDS
    _CUR_REQ_IDS = list(req_ids) if req_ids is not None else None


def _current_req_ids(K: int) -> list[str] | None:
    """Return the first K decode-segment request IDs, or None if not plumbed."""
    global _CUR_REQ_IDS
    if _CUR_REQ_IDS is None or len(_CUR_REQ_IDS) < K:
        return None
    return _CUR_REQ_IDS[:K]


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
