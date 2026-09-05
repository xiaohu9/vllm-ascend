"""T5 probe: capture the baseline (non-PIVOT) native top-k indices verbatim.

Design constraints (user-specified, 2026-09-05):
  * NO score re-computation. The baseline output is kept exactly as the op
    returns it -- capture only ``topk_indices`` (int32 positions), never an
    fp32 re-score. The baseline path is not touched; the op already produced
    the indices, we just record them.
  * ONE env gate (``VLLM_ASCEND_TOPK_PROBE``, default off). Dump dir, layer
    selector, TP rank and sampling constants are module-level constants here,
    not env vars (the earlier 4-var design was rejected as too many knobs).
  * The analyzer emits TEXT (report.json / report.txt), no PNG -- collection
    artifacts cannot be shipped off the work server, but text can.

Sampling model (mirrors pivot_indexer._dump_gate): only indexer layers ever
call in, layer selector is "first" or a layer-name substring, stride spreads
captures over the run. Per (req, step) we keep the 2048 int32 positions of one
representative query row of the group; the analyzer derives IoU / decay /
residency / heatmap statistics offline. Bounded memory: a small bounded queue +
a daemon writer thread; nothing is kept per step beyond the captured row.

Diagnostics only -- never raises, zero hot-path cost when the gate is off.
"""

from __future__ import annotations

import os
import queue
import threading

import torch

from vllm.logger import logger
from vllm_ascend import envs

# ---- module constants (not env vars) -------------------------------------
_DUMP_DIR = "/tmp/topk_probe"
# Layer selector: "first" pins the first indexer layer seen; otherwise an exact
# layer-name substring (e.g. "layers.7"). Only indexer layers reach capture.
_LAYER_SELECTOR = "first"
# TP rank that writes to disk. With SP/CP off, TP data is replicated across
# ranks, so a single rank suffices.
_WRITE_RANK = "0"
# Bounded sampling knobs.
_MAX_REQS = 8          # distinct requests tracked (first-seen, in step order)
_STRIDE = 1            # capture every Nth (layer, step) hit
_MAX_STEPS_PER_REQ = 256  # hard cap on queued records per request (memory bound)

# ---- module state (mirrors pivot_indexer._dump_*) ------------------------
_probe_layer = None  # pinned indexer layer (None until first call)
_probe_seen: list[str] = []
_probe_hits = 0
_probe_miss_warned = False
_step_req_ids: list[str] = []  # set per step by model_runner, batch order
_queue: "queue.Queue[tuple] | None" = None
_writer_thread = None
_writer_started = False


def _local_rank() -> str:
    """TP rank for write sharding (multi-rank writers must not collide)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:
        pass
    return os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))


def _layer_num(name: str) -> int | None:
    """Last integer run in a layer name, or None if it has no digits."""
    i = len(name) - 1
    while i >= 0 and not name[i].isdigit():
        i -= 1
    if i < 0:
        return None
    j = i
    while j >= 0 and name[j].isdigit():
        j -= 1
    return int(name[j + 1:i + 1])


def _write_worker():
    """Daemon writer: consumes (req_id, step, layer_name, positions) tuples and
    appends fixed-size records to per-request files. One synchronous disk I/O
    per record, off the hot path."""
    try:
        os.makedirs(_DUMP_DIR, exist_ok=True)
        while True:
            item = _queue.get()
            if item is None:  # sentinel from atexit flush
                break
            req_id, step, layer_name, positions = item
            # sanitize req_id for use as a file name (ids may contain colons)
            safe = req_id.replace(":", "_").replace("/", "_")
            path = os.path.join(_DUMP_DIR, f"{safe}__{layer_name}.bin")
            with open(path, "ab") as f:
                f.write(step.to_bytes(8, "little"))
                f.write(positions.tobytes())
    except Exception:
        logger.exception("PIVOT topk probe writer failed")


def _ensure_writer():
    global _queue, _writer_thread, _writer_started
    if _writer_started:
        return
    _writer_started = True
    _queue = queue.Queue(maxsize=4096)
    _writer_thread = threading.Thread(target=_write_worker, daemon=True)
    _writer_thread.start()


def _gate(layer_name: str) -> bool:
    """Decide whether to capture THIS (layer, step) invocation. Never raises.

    Mirrors pivot_indexer._dump_gate: one-layer filter + stride sampler, with
    the real indexer-layer roster tracked so a bad selector warns instead of
    silently capturing nothing.
    """
    global _probe_layer, _probe_seen, _probe_miss_warned, _probe_hits
    if not envs.VLLM_ASCEND_TOPK_PROBE:
        return False
    if _probe_layer is None:
        want = _LAYER_SELECTOR
        if want == "first" or want in layer_name:
            _probe_layer = layer_name
        elif want != "first":
            if layer_name not in _probe_seen:
                _probe_seen.append(layer_name)
                logger.info_once(
                    "PIVOT probe: LAYER_SELECTOR=%r not matched yet; indexer "
                    "layer %s present (real indexer layers so far: %s)",
                    want, layer_name, ", ".join(_probe_seen))
            tw = _layer_num(want)
            if (not _probe_miss_warned and tw is not None
                    and any(_layer_num(l) is not None and _layer_num(l) > tw
                            for l in _probe_seen)):
                _probe_miss_warned = True
                logger.warning(
                    "PIVOT probe: LAYER_SELECTOR=%r matches no indexer layer "
                    "(real ones seen: %s); falling back to %s",
                    want, ", ".join(_probe_seen), layer_name)
                _probe_layer = layer_name
    if _probe_layer != layer_name:
        return False
    _probe_hits += 1
    return (_probe_hits - 1) % _STRIDE == 0


def set_step_request_ids(req_ids: list[str]) -> None:
    """Model runner feeds the current batch's request ids (batch order, global
    unique strings) once per step before execution. Used to map captured
    rows back to real requests -- never inferred from position."""
    global _step_req_ids
    if envs.VLLM_ASCEND_TOPK_PROBE:
        _step_req_ids = list(req_ids)


def capture(sfa_impl, topk_indices: torch.Tensor) -> None:
    """Record one native indexer invocation's top-2048 indices verbatim.

    Called from indexer_select_post_process right after the op returns, on the
    two BF16 branches. ``topk_indices`` is [D, 1, 2048] int32 (TND decode).
    Keeps one representative query row per tracked request; queue + daemon
    writer bound memory. Never raises.
    """
    global _step_req_ids
    if not envs.VLLM_ASCEND_TOPK_PROBE:
        return
    try:
        if not _gate(sfa_impl.layer_name):
            return
        if not _step_req_ids:
            return  # no id mapping yet; never guess
        if _local_rank() != _WRITE_RANK:
            return
        _ensure_writer()
        if torch.npu.is_current_stream_capturing():
            return  # graph capture is unsafe for D2H + I/O; eager runs only

        # Capture the first query row (row0) of the first _MAX_REQS requests.
        # topk_indices rows are TND (decode: g rows per request); row0 of
        # request r sits at row r * g. Without per-request g here we take the
        # leading _MAX_REQS rows of the batch -- batch order == request order,
        # so row i belongs to request _step_req_ids[i]'s first group row.
        n = min(topk_indices.shape[0], _MAX_REQS, len(_step_req_ids))
        if n == 0:
            return
        step = _probe_hits
        rows = topk_indices[:n, 0, :].cpu()  # [n, 2048] int32 (one D2H)
        for i in range(n):
            req_id = _step_req_ids[i]
            if _queue.full():
                return  # bounded memory: drop rather than grow unboundedly
            _queue.put((req_id, step, sfa_impl.layer_name, rows[i].clone()))
    except Exception:
        # self-protect: a probe bug must never take down the server
        logger.exception("PIVOT topk probe capture failed")


def flush() -> None:
    """Drain the queue (call at process exit; atexit also registered)."""
    global _queue
    try:
        if _queue is not None:
            _queue.put(None)
            if _writer_thread is not None:
                _writer_thread.join(timeout=10)
    except Exception:
        pass


import atexit

atexit.register(flush)
