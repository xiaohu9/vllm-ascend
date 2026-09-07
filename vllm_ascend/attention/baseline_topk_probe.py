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
# Layer selector: comma-separated entries (e.g. "layers.1,layers.30,
# layers.60" for shallow/mid/deep). A layer is captured when it matches ANY
# entry. A bare-integer entry ("2,22,26") is an EXACT layer-number match
# ("22" hits model.layers.22.self_attn.attn but never layers.2x/222);
# anything else ("layers.38", "first") keeps substring semantics. "first"
# pins the very first indexer layer seen -- the default keeps the original
# single-layer behavior. Only indexer layers reach capture.
_LAYER_SELECTOR = "first"
# TP rank that writes to disk. With SP/CP off, TP data is replicated across
# ranks, so a single rank suffices.
_WRITE_RANK = "0"
# Bounded sampling knobs. Memory bound = bounded queue (maxsize) + request
# cap; there is deliberately no per-request step cap (the queue is the bound).
_MAX_REQS = 64         # distinct requests tracked (first-seen, in step order)
_STRIDE = 1            # capture every Nth (layer, step) hit

# ---- module state (mirrors pivot_indexer._dump_*) ------------------------
_probe_layers: set[str] = set()  # pinned indexer layers (fill as they match)
_probe_seen: list[str] = []
_probe_hits = 0
# Real decode step counter (incremented once per step by set_step_request_ids,
# which the model runner calls before each step's execution). All indexer
# layers in one step share the same value, so it is the cross-layer alignment
# key for per-step IoU (shallow vs mid vs deep) -- _probe_hits is global and
# cannot align layers.
_real_step = 0
_probe_miss_warned = False
_probe_map_warned = False  # once: topk rows could not be mapped to requests
_step_req_ids: list[str] = []  # set per step by model_runner, batch order
_queue: "queue.Queue[tuple] | None" = None
_writer_thread = None
_writer_started = False
_dump_cleaned = False  # once: wiped stale captures from a previous run


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


def _match(w: str, layer_name: str) -> bool:
    """Selector-entry match. Bare integers are EXACT layer-number matches
    (w="22" hits layer_name "model.layers.22.self_attn.attn" -> _layer_num 22,
    never "model.layers.222..." nor "model.layers.2x..."); any other entry
    ("layers.38", "attn", ...) keeps substring semantics. Never raises."""
    if w.isdigit():
        n = _layer_num(layer_name)
        return n is not None and n == int(w)
    return w in layer_name


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
            try:
                req_id, step, layer_name, positions = item
                # sanitize req_id for use as a file name (ids may contain colons)
                safe = req_id.replace(":", "_").replace("/", "_")
                path = os.path.join(_DUMP_DIR, f"{safe}__{layer_name}.bin")
                with open(path, "ab") as f:
                    f.write(step.to_bytes(8, "little"))
                    # positions is a CPU int32 tensor (rows[i].clone()); numpy()
                    # is the only byte-dump path -- torch.Tensor has no tobytes().
                    f.write(positions.cpu().numpy().tobytes())
            except Exception:
                # one bad record must not kill the writer thread and lose the
                # rest of the queue (that would silently empty the capture).
                logger.exception("PIVOT topk probe writer failed on one record")
    except Exception:
        logger.exception("PIVOT topk probe writer failed")


def _ensure_writer():
    global _queue, _writer_thread, _writer_started, _dump_cleaned
    if _writer_started:
        return
    _writer_started = True
    # Fresh dump per run: the writer APPENDS to per-request files, so captures
    # from a previous process (possibly a different model / layer selector)
    # would otherwise persist and mix into this run's report, looking like
    # layers the probe "self-selected" (e.g. 2/22/26/30 that the current model
    # never routes through capture). Wipe once at startup -- only the rank-0
    # writer reaches this point (rank check precedes _ensure_writer in
    # capture), so this cannot collide with another process.
    if not _dump_cleaned:
        _dump_cleaned = True
        try:
            if os.path.isdir(_DUMP_DIR):
                for fn in os.listdir(_DUMP_DIR):
                    if fn.endswith(".bin"):
                        os.remove(os.path.join(_DUMP_DIR, fn))
        except Exception:
            logger.exception("PIVOT topk probe: failed to clean old captures")
    _queue = queue.Queue(maxsize=4096)
    _writer_thread = threading.Thread(target=_write_worker, daemon=True)
    _writer_thread.start()


def _gate(layer_name: str) -> bool:
    """Decide whether to capture THIS (layer, step) invocation. Never raises.

    Mirrors pivot_indexer._dump_gate: LAYER_SELECTOR is a comma-separated list
    of entries, a layer is captured when it matches ANY of them (_match: bare
    integers are exact layer numbers, everything else substring), and "first"
    pins the very first indexer layer (so the default keeps single-layer
    behavior). The real indexer-layer roster is tracked so a selector naming no
    indexer layer warns instead of silently capturing nothing -- there is
    deliberately NO fallback pin (multi-layer selection must not drift onto
    unrequested layers).
    """
    global _probe_layers, _probe_seen, _probe_miss_warned, _probe_hits
    if not envs.VLLM_ASCEND_TOPK_PROBE:
        return False
    wants = [w.strip() for w in _LAYER_SELECTOR.split(",") if w.strip()]
    if any(w == "first" for w in wants) and not _probe_layers:
        _probe_layers.add(layer_name)  # "first": pin the first indexer layer
    if any(w != "first" and _match(w, layer_name) for w in wants):
        _probe_layers.add(layer_name)
    if layer_name not in _probe_layers:
        if layer_name not in _probe_seen:
            _probe_seen.append(layer_name)
            logger.info_once(
                "PIVOT probe: LAYER_SELECTOR=%r not matched yet; indexer "
                "layer %s present (real indexer layers so far: %s)",
                _LAYER_SELECTOR, layer_name, ", ".join(_probe_seen))
        explicit = [w for w in wants if w != "first"]
        if not _probe_miss_warned and explicit:
            tw = _layer_num(explicit[0])
            if (tw is not None
                    and any(_layer_num(l) is not None and _layer_num(l) > tw
                            for l in _probe_seen)):
                _probe_miss_warned = True
                logger.warning(
                    "PIVOT probe: LAYER_SELECTOR=%r matches no indexer layer "
                    "(real ones seen: %s)",
                    _LAYER_SELECTOR, ", ".join(_probe_seen))
        return False
    _probe_hits += 1
    return (_probe_hits - 1) % _STRIDE == 0


def set_step_request_ids(req_ids: list[str]) -> None:
    """Model runner feeds the current batch's request ids (batch order, global
    unique strings) once per step before execution. Used to map captured
    rows back to real requests -- never inferred from position. Also advances
    the real-step counter, the cross-layer alignment key."""
    global _step_req_ids, _real_step
    if envs.VLLM_ASCEND_TOPK_PROBE:
        _step_req_ids = list(req_ids)
        _real_step += 1


def _request_row_offsets(query_start_loc, n_requests: int, total_rows: int):
    """First-query-row offset per request (batch order) from the indexer's
    ``actual_seq_lengths_query`` (cumulative per-request query-row counts:
    [0, g, 2g, ..., Kg] with a leading 0), or None if it cannot be derived.
    With MTP g > 1, TND topk_indices carries g rows per request and row i
    belongs to request i // g -- request r's rows are [starts[r], starts[r+1])
    and its first query row sits at starts[r]. Never guesses: a malformed /
    misaligned tensor yields None, never a fabricated mapping."""
    if query_start_loc is None or n_requests <= 0:
        return None
    try:
        vals = [int(x) for x in query_start_loc.detach().cpu().tolist()]
    except Exception:
        return None
    if len(vals) == n_requests + 1 and vals[0] == 0:
        starts = vals[:-1]          # start-loc form (leading 0)
    elif len(vals) == n_requests and vals[0] != 0:
        starts = [0] + vals[:-1]    # cumulative ends without the leading 0
    else:
        return None
    if (len(starts) != n_requests or starts[0] != 0
            or any(b <= a for a, b in zip(starts, starts[1:]))
            or starts[-1] >= total_rows):
        return None
    return starts


def capture(sfa_impl, topk_indices: torch.Tensor, query_start_loc=None) -> None:
    """Record one native indexer invocation's top-2048 indices verbatim.

    Called from indexer_select_post_process right after the op returns, on the
    two BF16 branches. ``topk_indices`` is [D, 1, 2048] int32 (TND decode: g
    query rows per request). Keeps the first query row of each tracked request,
    labelled with its real request id; queue + daemon writer bound memory.
    Never raises.

    ``query_start_loc`` is the indexer's ``actual_seq_lengths_query`` (cumulative
    per-request query-row counts, [0, g, 2g, ...]) -- the only reliable way to
    map TND rows back to requests: with MTP g > 1 row i belongs to request
    i // g, NOT request i, so labelling rows by leading index mislabels every
    request past the first group (the pre-fix behaviour turned g=4 captures
    into "8 requests" that were really 2). When it is unavailable we fall back
    to a uniform g = D // K only on exact division, and otherwise skip the step
    rather than guess a mapping.
    """
    global _step_req_ids, _real_step, _probe_map_warned
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

        n_reqs = min(len(_step_req_ids), _MAX_REQS)
        if n_reqs == 0:
            return
        step = _real_step  # real decode step: same value for every layer in one step
        starts = _request_row_offsets(query_start_loc, len(_step_req_ids),
                                      topk_indices.shape[0])
        if starts is None and topk_indices.shape[0] % len(_step_req_ids) == 0:
            # no per-request lens, but exact division implies uniform MTP decode
            g = topk_indices.shape[0] // len(_step_req_ids)
            starts = [r * g for r in range(len(_step_req_ids))]
        if starts is None:
            if not _probe_map_warned:
                _probe_map_warned = True
                logger.warning(
                    "PIVOT probe: cannot map topk rows to requests "
                    "(D=%d, K=%d) -- skipping this step",
                    topk_indices.shape[0], len(_step_req_ids))
            return  # never guess a mapping
        # row-0 only: with MTP OFF (g=1) each decode step is exactly one token,
        # so the adjacent-step IoU computed offline IS the adjacent-token IoU.
        # (To compare parallelisms, re-run the probe under different g rather
        # than capturing all rows -- keeps this capture single-row & minimal.)
        row_idx = torch.tensor([starts[r] for r in range(n_reqs)],
                               dtype=torch.long, device=topk_indices.device)
        rows = topk_indices.index_select(0, row_idx)[:, 0, :].cpu()
        # [n_reqs, 2048] int32 -- one D2H; row r == request r's first query row
        for i in range(n_reqs):
            if _queue.full():
                return  # bounded memory: drop rather than grow unboundedly
            _queue.put((_step_req_ids[i], step, sfa_impl.layer_name,
                        rows[i].clone()))
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
