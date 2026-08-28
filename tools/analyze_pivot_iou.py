#!/usr/bin/env python3
"""Offline analyzer for the PIVOT step-to-step top-k change probe.

Reads the .npz files produced by vllm_ascend/attention/pivot_iou_probe.py.

The probe does NOT store the big coarse/refine arrays. For each SFA layer and
each adjacent pair of decode steps it records only the symmetric-difference
counts of the top-k index sets:

  * coarse_diff = |C(t) Delta C(t+1)| per matched request, summed
  * refine_diff = |R(t) Delta R(t+1)| per matched query-row, summed

This answers the innovation-#1 premise directly: how much of the candidate
pool actually changes between adjacent decode steps. Small diff counts vs the
4096/2048 budget mean the pool is temporally stable -> worth reusing across
steps.

Pure CPU numpy -- no torch, no NPU. Run on any host with numpy.

Usage:
  python tools/analyze_pivot_iou.py /tmp/pivot_iou/ --out report.json --plot report.png
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Budgets the probe compares against (doc: coarse 4096, refine 2048).
_COARSE_BUDGET = 4096
_REFINE_BUDGET = 2048


def _load_rows(dir_path: str) -> list[dict]:
    rows: list[dict] = []
    names = sorted(f for f in os.listdir(dir_path) if f.endswith(".npz"))
    if not names:
        raise SystemExit(f"no .npz files under {dir_path}")
    for name in names:
        with np.load(os.path.join(dir_path, name), allow_pickle=True) as z:
            for r in z["rows"].tolist():
                rows.append(r)
    return rows


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.array(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def _verdict(coarse: dict, refine: dict) -> str:
    """Map measured diff counts to the engineering premise (doc §4)."""
    cm = coarse.get("median", -1)
    lines = []
    if cm < 0:
        lines.append("no coarse records -- nothing to judge")
    elif cm <= _COARSE_BUDGET * 0.3:
        lines.append(f"coarse_diff median={cm:.0f} <= 30% of 4096: pool highly "
                     "overlapping -> persistent-pool premise HOLDS, design "
                     "incremental refresh")
    elif cm <= _COARSE_BUDGET * 0.6:
        lines.append(f"coarse_diff median={cm:.0f} in (30%,60%] of 4096: "
                     "moderate overlap -> pool feasible but refresh period "
                     "must be short")
    else:
        lines.append(f"coarse_diff median={cm:.0f} > 60% of 4096: premise "
                     "REFUTED -> downgrade innovation #1 to short-window cache")
    rm = refine.get("median", -1)
    if rm >= 0:
        frac = rm / _REFINE_BUDGET
        if frac <= 0.3:
            lines.append(f"refine_diff median={rm:.0f} <= 30% of 2048: refine "
                         "sets stable -> P1 loss dominated by coarse-screen "
                         "misses, not refine churn")
    return "; ".join(lines)


def _plot(report: dict, out_base: str) -> None:
    """Render coarse/refine diff-count distributions to a single PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PIVOT adjacent-step top-k change (GLM-5.2-w4a8 BF16, A3)",
                 fontsize=13)
    panels = [
        ("coarse_diff", "coarse set change |A△B| (budget 4096)", _COARSE_BUDGET, axes[0]),
        ("refine_diff", "refine set change |A△B| (budget 2048)", _REFINE_BUDGET, axes[1]),
    ]
    for key, title, budget, ax in panels:
        st = report[key]
        if not st.get("n"):
            ax.set_title(title + "\n(no data)")
            continue
        raw = st["raw"]
        ax.hist(raw, bins=40, alpha=0.7, color="#4c72b0")
        for stat, color in (("median", "red"), ("p10", "gray"), ("p90", "gray")):
            v = st.get(stat)
            if v is not None:
                ax.axvline(v, color=color, ls="--", lw=1.2,
                           label=f"{stat}={v:.0f}")
        ax.axvline(budget * 0.3, color="green", ls=":", lw=1,
                   label="30% budget")
        ax.set_title(f"{title} (n={st['n']}, mean={st.get('mean', float('nan')):.0f})")
        ax.set_xlabel("count of changed indices")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_base + ".png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", help="directory holding pivot_iou_*.npz")
    ap.add_argument("--out", default=None, help="write report JSON here")
    ap.add_argument("--plot", default=None, metavar="PNG",
                    help="also render plots to PNG (default: <--out>.png)")
    args = ap.parse_args()

    rows = _load_rows(args.dir)
    print(f"loaded {len(rows)} (layer, step) records")
    if not rows:
        raise SystemExit("nothing to analyze")

    coarse_all: list[float] = []     # per-request coarse diff
    refine_per_req: list[float] = []  # per-request refine diff (sum over g rows)
    g_counts: dict[int, int] = {}
    tot_K = 0   # prev requests available to match
    tot_matched = 0  # requests that were uniquely matched
    per_layer: dict[str, dict] = {}

    for r in rows:
        layer = r["layer"]
        g = int(r.get("g", 0))
        n_req = max(1, r.get("n_req", 1))
        tot_K += int(r.get("K", 0))
        tot_matched += int(r.get("n_req", 0))
        if g > 0:
            g_counts[g] = g_counts.get(g, 0) + 1
        pl = per_layer.setdefault(layer, {"coarse": [], "refine_per_req": []})
        if "coarse_diff" in r:
            # coarse_diff summed over matched requests; normalize per request
            v = r["coarse_diff"] / n_req
            coarse_all.append(v)
            pl["coarse"].append(v)
        if "refine_diff" in r:
            # refine_diff = sum over g rows AND over matched requests; keep
            # per-request (sum over rows) so per-row = this / g.
            v = r["refine_diff"] / n_req
            refine_per_req.append(v)
            pl["refine_per_req"].append(v)

    coarse_stats = _stats(coarse_all)
    refine_per_req_stats = _stats(refine_per_req)

    # Derive per-row refine diff from the dominant g (or median g of records).
    g_mode = max(g_counts, key=g_counts.get) if g_counts else None
    if g_mode:
        per_row = [v / g_mode for v in refine_per_req]
        refine_per_row_stats = _stats(per_row)
        # IoU implied by the median diff vs the per-row budget (2048): if the
        # measured budget differs this is only approximate -- g/budget are the
        # two unknowns the probe must confirm.
        rd = refine_per_row_stats.get("median", -1)
        if rd >= 0 and rd <= _REFINE_BUDGET:
            refine_iou = (_REFINE_BUDGET * 2 - rd) / (_REFINE_BUDGET * 2 + rd)
        else:
            refine_iou = None  # diff exceeds budget -> budget/g mismatch, trust nothing
    else:
        refine_per_row_stats = {"n": 0}
        refine_iou = None

    # IoU implied by the median coarse diff (per-request, budget 4096).
    cd = coarse_stats.get("median", -1)
    coarse_iou = (_COARSE_BUDGET * 2 - cd) / (_COARSE_BUDGET * 2 + cd) if 0 <= cd <= _COARSE_BUDGET else None

    report = {
        "config": {
            "g_counts": g_counts,
            "g_mode": g_mode,
            "coarse_budget": _COARSE_BUDGET,
            "refine_budget": _REFINE_BUDGET,
        },
        # Match quality: what fraction of prev requests could be uniquely
        # matched by exact request-id (the probe pairs by real req id string,
        # never by heuristic). Low (<0.5) => batch churns a lot between steps;
        # the diff numbers are then based on a small, possibly biased sample
        # and should be read with caution.
        "matching": {
            "prev_requests": tot_K,
            "matched_requests": tot_matched,
            "match_rate": round(tot_matched / tot_K, 4) if tot_K else None,
        },
        "coarse_diff_per_req": coarse_stats,
        "coarse_median_iou": coarse_iou,
        "refine_diff_per_req": refine_per_req_stats,
        "refine_diff_per_row": refine_per_row_stats,
        "refine_median_iou": refine_iou,
        "per_layer": {
            k: {"coarse_diff_per_req": _stats(v["coarse"]),
                "refine_diff_per_req": _stats(v["refine_per_req"])}
            for k, v in per_layer.items()
        },
        "verdict": _verdict(coarse_stats, refine_per_req_stats),
    }

    out_base = None
    if args.plot:
        out_base = args.plot[:-4] if args.plot.endswith(".png") else args.plot
    elif args.out:
        out_base = args.out[:-5] if args.out.endswith(".json") else args.out
    if out_base:
        # raw samples ride along only for the histogram, then stripped.
        # refine panel uses per-row diff (÷g) so the "budget 2048" axis holds.
        refine_plot = per_row if g_mode else refine_per_req
        _plot({"coarse_diff": dict(_stats(coarse_all), raw=coarse_all),
               "refine_diff": dict(_stats(refine_plot), raw=refine_plot)}, out_base)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
