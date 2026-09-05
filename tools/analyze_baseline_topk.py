"""Analyze baseline topk probe captures -> TEXT report (no figures).

The probe (vllm_ascend/attention/baseline_topk_probe.py) writes fixed-size
records to ``{dump_dir}/{req_id}__{layer}.bin``: each record is
step (int64 LE) + 2048 x int32 LE positions (8200 bytes).

This analyzer reads all captures and emits:
  * report.json  -- full structured statistics (enough to re-plot locally)
  * report.txt   -- human-readable summary with decision-line comparisons

No PNG / matplotlib output: collection artifacts cannot be shipped off the
work server, but text can -- plots are rebuilt locally from the JSON numbers.

Usage:
  python tools/analyze_baseline_topk.py <dump_dir> --out report.json --txt report.txt
"""

from __future__ import annotations

import argparse
import json
import os
import struct

RECORD_BYTES = 8 + 2048 * 4
TOP_K = 2048


def _read_records(path: str) -> list[tuple[int, list[int]]]:
    """[(step, positions), ...] from one per-request-per-layer capture file."""
    records = []
    with open(path, "rb") as f:
        while True:
            header = f.read(RECORD_BYTES)
            if len(header) < RECORD_BYTES:
                break
            step = struct.unpack("<q", header[:8])[0]
            positions = list(struct.unpack("<%di" % TOP_K, header[8:]))
            records.append((step, positions))
    records.sort(key=lambda r: r[0])
    return records


def _iou(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0


def _analyze_series(records: list[tuple[int, list[int]]], max_dist: int = 16):
    """M2 adjacent IoU, M3 decay curve, M4 retention, M5 streak stats."""
    if len(records) < 2:
        return None
    iou1 = [_iou(records[t][1], records[t + 1][1]) for t in range(len(records) - 1)]
    decay = {}
    for d in range(1, min(max_dist, len(records)) + 1):
        pairs = [_iou(records[t][1], records[t + d][1]) for t in range(len(records) - d)]
        decay[d] = (sum(pairs) / len(pairs)) if pairs else 1.0
    # M4 retention: fraction of the step-0 set still present at distance d.
    s0 = set(records[0][1])
    retention = {}
    for d in range(1, min(max_dist, len(records)) + 1):
        if len(s0) == 0:
            retention[d] = 1.0
        else:
            sd = set(records[d][1])
            retention[d] = len(s0 & sd) / len(s0)
    # M5 streak: longest consecutive-run length per position over the series.
    from collections import defaultdict
    streaks: dict[int, int] = defaultdict(int)
    cur: dict[int, int] = {}
    for _, positions in records:
        ps = set(positions)
        for pos in cur:
            if pos not in ps:
                streaks[pos] = max(streaks[pos], cur[pos])
        nxt = {pos: cur.get(pos, 0) + 1 for pos in ps}
        cur = nxt
    for pos, c in cur.items():
        streaks[pos] = max(streaks[pos], c)
    vals = sorted(streaks.values())
    n = len(vals)
    median = vals[n // 2] if n else 0
    mean = sum(vals) / n if n else 0.0
    p90 = vals[int(n * 0.9) - 1] if n else 0
    return {
        "n_steps": len(records),
        "iou1_mean": sum(iou1) / len(iou1),
        "iou1_min": min(iou1),
        "iou1_median": sorted(iou1)[len(iou1) // 2],
        "decay": {str(d): round(v, 4) for d, v in decay.items()},
        "retention": {str(d): round(v, 4) for d, v in retention.items()},
        "streak_mean": round(mean, 2),
        "streak_median": median,
        "streak_p90": p90,
        "streak_max": vals[-1] if vals else 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump_dir")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--txt", default="report.txt")
    ap.add_argument("--max-dist", type=int, default=16)
    args = ap.parse_args()

    files = sorted(
        f for f in os.listdir(args.dump_dir) if f.endswith(".bin")
    ) if os.path.isdir(args.dump_dir) else []
    if not files:
        print(f"no captures found under {args.dump_dir}")
        return 1

    per_req: dict[str, dict[str, dict]] = {}
    for fn in files:
        base = fn[:-4]
        req_id, _, layer = base.rpartition("__")
        recs = _read_records(os.path.join(args.dump_dir, fn))
        if len(recs) < 2:
            continue
        per_req.setdefault(req_id, {})[layer] = _analyze_series(recs, args.max_dist)

    # aggregate across requests per layer
    layers: dict[str, list] = {}
    for req_data in per_req.values():
        for layer, stat in req_data.items():
            layers.setdefault(layer, []).append(stat)

    layer_agg = {}
    for layer, stats in layers.items():
        good = [s for s in stats if s]
        layer_agg[layer] = {
            "n_requests": len(good),
            "iou1_mean": round(sum(s["iou1_mean"] for s in good) / len(good), 4),
            "iou1_min": min(s["iou1_min"] for s in good),
            "streak_mean": round(sum(s["streak_mean"] for s in good) / len(good), 2),
            "decay_mean": {str(d): round(sum(s["decay"][str(d)] for s in good) / len(good), 4) for d in range(1, args.max_dist + 1) if good and str(d) in good[0]["decay"]},
        }

    report = {
        "top_k": TOP_K,
        "n_layers": len(layers),
        "n_requests_total": len(per_req),
        "per_request": per_req,
        "per_layer": layer_agg,
        "decision": {
            "iou1_ge_0.7": all(l["iou1_mean"] >= 0.7 for l in layer_agg.values()),
            "note": "IoU(1)>=0.7 -> incremental maintenance viable; decay slope ~0 -> high residency.",
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(args.txt, "w", encoding="utf-8") as f:
        f.write("=== PIVOT baseline topk probe report (text; plots rebuilt locally from report.json) ===\n\n")
        f.write(f"top_k={TOP_K}  layers={len(layers)}  requests={len(per_req)}\n\n")
        for layer, agg in layer_agg.items():
            f.write(f"[layer {layer}] requests={agg['n_requests']}\n")
            f.write(f"  adjacent-step IoU mean={agg['iou1_mean']:.4f} (min {agg['iou1_min']:.4f})  "
                    f"decision: {'PASS >=0.7' if agg['iou1_mean'] >= 0.7 else 'FAIL <0.7'}\n")
            f.write(f"  streak mean={agg['streak_mean']:.2f}\n")
            f.write(f"  decay: " + ", ".join(f"d{d}={v:.3f}" for d, v in sorted(agg["decay_mean"].items(), key=lambda kv: int(kv[0]))) + "\n")
            f.write("\n")
        f.write("notes: M7 (in-group heatmap) / M8 (cross-layer) / M10 (window hit) "
                "require probe extensions; current probe captures row0 single layer.\n")
    print(f"wrote {args.out} and {args.txt}")


if __name__ == "__main__":
    raise SystemExit(main())
