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

    coarse_all: list[float] = []
    refine_all: list[float] = []
    per_layer: dict[str, dict] = {}

    for r in rows:
        layer = r["layer"]
        pl = per_layer.setdefault(layer, {"coarse": [], "refine": []})
        if "coarse_diff" in r:
            # coarse_diff summed over matched requests; normalize per request
            v = r["coarse_diff"] / max(1, r.get("n_req", 1))
            coarse_all.append(v)
            pl["coarse"].append(v)
        if "refine_diff" in r:
            v = r["refine_diff"] / max(1, r.get("n_req", 1))
            refine_all.append(v)
            pl["refine"].append(v)

    report = {
        "coarse_diff": _stats(coarse_all),
        "refine_diff": _stats(refine_all),
        "per_layer": {
            k: {"coarse_diff": _stats(v["coarse"]),
                "refine_diff": _stats(v["refine"])}
            for k, v in per_layer.items()
        },
        "verdict": _verdict(_stats(coarse_all), _stats(refine_all)),
    }

    out_base = None
    if args.plot:
        out_base = args.plot[:-4] if args.plot.endswith(".png") else args.plot
    elif args.out:
        out_base = args.out[:-5] if args.out.endswith(".json") else args.out
    if out_base:
        # raw samples ride along only for the histogram, then stripped
        rp = report
        _plot({"coarse_diff": dict(_stats(coarse_all), raw=coarse_all),
               "refine_diff": dict(_stats(refine_all), raw=refine_all)}, out_base)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
