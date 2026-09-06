"""Extra experiments derived from the SAME baseline topk captures (no re-probe).

Reads the same {dump_dir}/{req_id}__{layer}.bin captures as
analyze_baseline_topk.py (record = int64 step + 2048 x int32 positions), then
derives claims the base report does not cover:

  * cross-layer IoU    -- at the SAME decode step, how similar are the top-2048
                          sets of layer i vs layer j? Proves whether cross-layer
                          candidate sharing is viable (PIVOT's cross-layer reuse
                          premise). Requires the probe to have captured the same
                          request at multiple layers at the same steps; aligned
                          steps are intersected by exact step number.
  * core/transient     -- fraction of positions present in >=90% of steps
                          ("core"/anchors), >=50% ("stable"), <10% ("transient").
                          Quantifies "few anchors last the whole run".
  * streak histogram   -- fraction of positions whose longest consecutive run
                          is >= {1,5,16,64,256} steps, and == full length.
  * churn size         -- mean |A_t \\ A_{t+1}| (new entries per step; equals
                          evicted count because both sets have 2048 elements).

Output: report_extra.json + report_extra.txt (+ --png 2x2 figure). Matplotlib is
optional; without it the script degrades to text (safe on the NPU server).

Usage:
  python tools/analyze_baseline_topk_extra.py <dump_dir> \
      --out report_extra.json --txt report_extra.txt [--png report_extra.png]
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


def _iou(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def _streak_hist(records: list[tuple[int, list[int]]]) -> dict[str, float]:
    """Fractions of positions whose longest consecutive run >= thresholds."""
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
    n = len(streaks)
    full = len(records)
    if n == 0:
        return {}
    out = {}
    for thr in (1, 5, 16, 64, 256):
        out[f"ge_{thr}"] = round(sum(1 for v in streaks.values() if v >= thr) / n, 4)
    out["ge_full"] = round(sum(1 for v in streaks.values() if v >= full) / n, 4)
    return out


def _layer_stability(records: list[tuple[int, list[int]]]) -> dict:
    """core/stable/transient fractions + churn size, per (req, layer)."""
    if len(records) < 2:
        return None
    step_sets = [set(pos) for _, pos in records]
    n = len(step_sets)
    counts: dict[int, int] = {}
    for s in step_sets:
        for pos in s:
            counts[pos] = counts.get(pos, 0) + 1
    npos = len(counts)
    if npos == 0:
        return None
    core = sum(1 for c in counts.values() if c / n >= 0.90) / npos
    stable = sum(1 for c in counts.values() if c / n >= 0.50) / npos
    transient = sum(1 for c in counts.values() if c / n < 0.10) / npos
    new_entries = [len(step_sets[t] - step_sets[t + 1])
                   for t in range(n - 1)]
    return {
        "n_steps": n,
        "core_frac": round(core, 4),      # present in >=90% of steps (anchors)
        "stable_frac": round(stable, 4),  # >=50%
        "transient_frac": round(transient, 4),  # <10% (one-shot visitors)
        "churn_new_per_step": round(sum(new_entries) / len(new_entries), 2),
        "streak": _streak_hist(records),
    }


def _cross_layer(per_req_layer: dict[str, dict[str, list]]) -> dict:
    """Per (req, layer-pair): mean IoU over steps aligned exactly by number."""
    pairs: dict[str, dict] = {}
    for layer_recs in per_req_layer.values():
        layers = sorted(layer_recs)
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                la, lb = layers[i], layers[j]
                key = f"{la}__{lb}"
                by_step = {
                    "a": {s: set(p) for s, p in layer_recs[la]},
                    "b": {s: set(p) for s, p in layer_recs[lb]},
                }
                aligned = sorted(set(by_step["a"]) & set(by_step["b"]))
                if not aligned:
                    pairs.setdefault(key, {"n_pairs": 0, "iou_mean": None,
                                           "iou_min": None, "note": "steps not aligned across layers"})
                    continue
                ious = [_iou(by_step["a"][s], by_step["b"][s]) for s in aligned]
                p = pairs.setdefault(key, {"n_pairs": 0, "iou_mean": None,
                                           "iou_min": None})
                p["n_pairs"] += len(ious)
                # running accumulate
                cur = p.get("_acc")
                if cur is None:
                    cur = []
                cur.extend(ious)
                p["_acc"] = cur
                p["iou_mean"] = round(sum(cur) / len(cur), 4)
                p["iou_min"] = round(min(cur), 4)
    for key, p in list(pairs.items()):
        p.pop("_acc", None)
    return {"layer_pairs": sorted(pairs), "pairs": pairs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir")
    ap.add_argument("--out", default="report_extra.json")
    ap.add_argument("--txt", default="report_extra.txt")
    ap.add_argument("--png", default="", help="optional 2x2 figure")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.dump_dir)
                   if f.endswith(".bin")) if os.path.isdir(args.dump_dir) else []
    if not files:
        print(f"no captures found under {args.dump_dir}")
        return 1

    # req -> layer -> [(step, positions), ...] (only requests with >=2 steps)
    per_req: dict[str, dict[str, list]] = {}
    for fn in files:
        base = fn[:-4]
        req_id, _, layer = base.rpartition("__")
        recs = _read_records(os.path.join(args.dump_dir, fn))
        if len(recs) >= 2:
            per_req.setdefault(req_id, {})[layer] = recs

    # per-layer aggregates over requests
    layer_stab: dict[str, dict] = {}
    for req_data in per_req.values():
        for layer, recs in req_data.items():
            s = _layer_stability(recs)
            if s:
                agg = layer_stab.setdefault(layer, {"n_requests": 0, "_sum": {}})
                agg["n_requests"] += 1
                for k, v in s.items():
                    if isinstance(v, dict):  # streak hist -> accumulate per thr
                        sub = agg["_sum"].setdefault(k, {})
                        for t, val in v.items():
                            sub[t] = sub.get(t, 0.0) + val
                    else:
                        agg["_sum"][k] = agg["_sum"].get(k, 0.0) + v
    for layer, agg in layer_stab.items():
        nr = agg["n_requests"]
        agg.update({k: round(v / nr, 4) for k, v in agg["_sum"].items()
                    if not isinstance(v, dict)})
        for k, v in agg["_sum"].items():
            if isinstance(v, dict):
                agg[k] = {t: round(val / nr, 4) for t, val in v.items()}
        del agg["_sum"]

    xl = _cross_layer(per_req)
    report = {
        "top_k": TOP_K,
        "n_requests_total": len(per_req),
        "n_layers": len(layer_stab),
        "cross_layer": xl,
        "stability": layer_stab,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "=== PIVOT baseline topk EXTRA (same captures, new claims; no re-probe) ===",
        f"top_k={TOP_K}  requests={len(per_req)}  layers={len(layer_stab)}",
        "",
        "-- cross-layer IoU (same decode step, layer i vs j): top-2048 跨层相似度 --",
        "   高 => 跨层候选共享可行;低 => 各层候选彼此独立,共享收益有限。",
    ]
    for key in xl["layer_pairs"]:
        p = xl["pairs"][key]
        if p["iou_mean"] is None:
            lines.append(f"  {key}: {p['note']}")
        else:
            lines.append(f"  {key}: mean={p['iou_mean']:.4f}  min={p['iou_min']:.4f}  "
                         f"(aligned steps x{p['n_pairs']})")
    lines.append("")
    lines.append("-- stability per layer (anchors vs transients, churn size) --")
    lines.append("   core>=90%步, stable>=50%, transient<10%, churn=每步换新数, "
                 "streak=最长连续段>=阈值的占比")
    for layer, s in sorted(layer_stab.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 1e9):
        st = s.get("streak", {})
        lines.append(
            f"  L{layer}: core={s['core_frac']:.2f} stable={s['stable_frac']:.2f} "
            f"transient={s['transient_frac']:.2f} churn/step={s['churn_new_per_step']:.0f} "
            f"| streak>=5:{st.get('ge_5', float('nan')):.2f} >=16:{st.get('ge_16', float('nan')):.2f} "
            f"==full:{st.get('ge_full', float('nan')):.2f}")
    text = "\n".join(lines)
    print(text)
    with open(args.txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    if args.png:
        try:
            _draw(layer_stab, xl, args.png)
        except ImportError as e:
            print(f"\n[PNG skipped: matplotlib missing ({e})]")
    return 0


def _draw(stab: dict[str, dict], xl: dict, out: str) -> None:
    """2x2 figure: (a) cross-layer IoU, (b) core/transient, (c) streak, (d) churn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cjk = any(w.lower() in f.name.lower() for f in
              __import__("matplotlib.font_manager", fromlist=["fontManager"]).fontManager.ttflist
              for w in ("Microsoft YaHei", "SimHei", "PingFang SC",
                        "Noto Sans CJK SC", "WenQuanYi", "Source Han Sans"))
    if cjk:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                           "PingFang SC", "Noto Sans CJK SC"]
        plt.rcParams["axes.unicode_minus"] = False
    cap = lambda zh, en: zh if cjk else en  # noqa: E731

    layers = sorted(stab, key=int)
    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.6))
    fig.suptitle(cap("PIVOT 基线 topk 追加实验(同一批探针数据,无重跑)",
                     "PIVOT baseline topk EXTRA (same captures, no re-probe)"),
                 fontsize=12, fontweight="bold")

    # (a) cross-layer IoU: bar per layer pair
    ax = axs[0, 0]
    keys = xl["layer_pairs"]
    vals = [xl["pairs"][k]["iou_mean"] for k in keys]
    cols = ["#1e7d46" if v and v >= 0.5 else "#c0392b" for v in vals]
    ax.bar(range(len(keys)), vals, color=cols, width=0.6, alpha=0.9)
    for i, v in enumerate(vals):
        if v is not None:
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("__", " vs ") for k in keys], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(cap("(a) 跨层 IoU(同时刻 top-2048 重合)\n高=跨层候选共享可行",
                     "(a) cross-layer IoU at the same step\nhigh = cross-layer sharing viable",
                     ), fontsize=9.5, pad=7)
    ax.grid(axis="y", alpha=0.25)

    # (b) core/stable/transient stacked bars per layer
    ax = axs[0, 1]
    x = range(len(layers))
    ax.bar(x, [stab[l]["core_frac"] for l in layers], 0.5, label="core ≥90%",
           color="#2f9e44")
    ax.bar(x, [stab[l]["stable_frac"] - stab[l]["core_frac"] for l in layers], 0.5,
           bottom=[stab[l]["core_frac"] for l in layers], label="stable 50–90%",
           color="#e8590c", alpha=0.8)
    ax.bar(x, [stab[l]["transient_frac"] for l in layers], 0.5,
           bottom=[stab[l]["stable_frac"] for l in layers], label="transient <10%",
           color="#adb5bd")
    ax.set_xticks(list(x)); ax.set_xticklabels(["L" + l for l in layers])
    ax.set_ylim(0, 1.05)
    ax.set_title(cap("(b) 位置驻留分层:锚点 vs 瞬态\n(按出现步数占比)",
                     "(b) position residency: anchors vs transients\n(by fraction of steps present)",
                     ), fontsize=9.5, pad=7)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    # (c) streak histogram: fraction of positions with longest run >= threshold
    ax = axs[1, 0]
    thrs = [5, 16, 64, 256, "full"]
    for l in layers:
        st = stab[l]["streak"]
        vals = [st.get("ge_%d" % t) if t != "full" else st.get("ge_full")
                for t in thrs]
        ax.plot(range(len(thrs)), vals, marker="o", ms=3, lw=1.5, label="L" + l)
    ax.set_xticks(range(len(thrs)))
    ax.set_xticklabels([f"≥{t}" if t != "full" else "==全长" for t in thrs])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(cap("位置占比", "fraction of positions"))
    ax.set_title(cap("(c) 最长驻留段分布:越陡=越多人短驻留\n(右端==全长=全程锚点)",
                     "(c) longest-run distribution: steep = many short stays\n(right end == full run = anchors)",
                     ), fontsize=9.5, pad=7)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (d) churn size per layer
    ax = axs[1, 1]
    vals = [stab[l]["churn_new_per_step"] for l in layers]
    ax.bar(x, vals, 0.5, color="#3b5bdb", alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 5, f"{v:.0f}", ha="center", fontsize=8.5)
    ax.set_xticks(list(x)); ax.set_xticklabels(["L" + l for l in layers])
    ax.set_ylabel(cap("每步换新位置数", "new entries per step"))
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title(cap("(d) 每步换新量(=驱逐量)\n2048 槽里每步换掉多少",
                     "(d) churn size: new entries per step\n(== evictions; 2048 slots total)",
                     ), fontsize=9.5, pad=7)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[saved {out}]")


if __name__ == "__main__":
    raise SystemExit(main())
