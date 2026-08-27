#!/usr/bin/env python3
"""Offline analyzer for PIVOT step-to-step IoU probe captures.

Reads the .npz files produced by vllm_ascend/attention/pivot_iou_probe.py
and computes the four IoU families defined in
plans/pivot_step_iou_measurement.md:

  * IoU_coarse   coarse set (4096) overlap between adjacent decode steps,
                 per request, within a layer
  * IoU_refine   refine set (2048) overlap between adjacent decode steps,
                 per query-row (same request, same row index i in the group)
  * IoU_group    refine-set overlap among the g query rows of one request
                 (MTP group internal consistency, innovation #4 premise)
  * IoU_d(d)     refine IoU vs step distance d (persistence-pool refresh
                 period)

Cross-step request alignment: within a layer, request r at step t has query
start offset s (cum_start). A live request at step t+1 has cum_start ==
s + g (each decode step appends g query rows). Chains are matched by this
exact equality; requests that finish / new requests that reset to 0 do not
match and are dropped (no false pairing).

Pure CPU numpy -- no torch, no NPU. Run on any host with numpy.

Usage:
  python tools/analyze_pivot_iou.py /tmp/pivot_iou/ --out report.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


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


def _pairwise_adjacent(rows: list[dict]):
    """Group rows by layer, yield (t, t+1) pairs with aligned requests.

    Within one layer, rows are ordered by step. Yields dicts:
      {layer, g, t, coarse: [(A, B)], refine: [(A, B)], ...}
    where A/B are numpy int arrays of a matched request (A = step t,
    B = step t+1), refine paired per row index i.
    """
    by_layer: dict[str, list[dict]] = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r)
    for layer, lrows in by_layer.items():
        lrows.sort(key=lambda r: r["step"])
        for a, b in zip(lrows, lrows[1:]):
            if a["step"] + 1 != b["step"] or a["g"] != b["g"]:
                continue  # flushed across npz boundaries / g changed
            g = a["g"]
            # map request -> row index by cum_start
            a_idx = {int(s): i for i, s in enumerate(a["cum_start"])}
            b_idx = {int(s): i for i, s in enumerate(b["cum_start"])}
            coarse_pairs: list[tuple] = []
            refine_pairs: list[tuple] = []
            for s, ia in a_idx.items():
                ib = b_idx.get(s + g)
                if ib is None:
                    continue
                coarse_pairs.append((a["coarse"][ia], b["coarse"][ib]))
                ra = a["refine"][ia * g:(ia + 1) * g]
                rb = b["refine"][ib * g:(ib + 1) * g]
                if ra.shape[0] == g and rb.shape[0] == g:
                    for i in range(g):
                        refine_pairs.append((ra[i], rb[i]))
            yield {"layer": layer, "g": g, "t": a["step"],
                   "coarse": coarse_pairs, "refine": refine_pairs,
                   "seq_lens_a": a["seq_lens"], "seq_lens_b": b["seq_lens"]}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard IoU of two 1-D position sets (values, order-independent)."""
    av = set(int(x) for x in a if x >= 0)
    bv = set(int(x) for x in b if x >= 0)
    if not av and not bv:
        return 1.0
    inter = len(av & bv)
    union = len(av | bv)
    return inter / union if union else 1.0


def _stats(vals: list[float], with_raw: bool = False) -> dict:
    if not vals:
        return {"n": 0}
    a = np.array(vals, dtype=np.float64)
    out = {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }
    if with_raw:
        out["raw"] = vals
    return out


def _verdict(coarse: dict, refine: dict, group: dict) -> str:
    """Map measured IoU to the engineering premise (doc §4 judgment lines)."""
    cm = coarse.get("median", -1)
    lines = []
    if cm < 0:
        lines.append("no coarse pairs -- nothing to judge")
    elif cm >= 0.7:
        lines.append("IoU_coarse median>=0.7: coarse sets highly overlapping -> "
                     "persistent-pool premise HOLDS, design incremental refresh")
    elif cm >= 0.4:
        lines.append("IoU_coarse median in [0.4,0.7): moderate overlap -> pool "
                     "feasible but refresh period must be short")
    else:
        lines.append("IoU_coarse median<0.4: premise REFUTED -> downgrade "
                     "innovation #1 to short-window cache or drop")
    rm = refine.get("median", -1)
    if rm >= 0.7:
        lines.append("IoU_refine median>=0.7: refine sets stable -> P1 loss is "
                     "dominated by coarse-screen misses, not refine churn")
    gm = group.get("median", -1)
    if gm >= 0.6:
        lines.append("IoU_group median>=0.6: MTP group rows near-identical -> "
                     "cross-group shared candidates (innovation #4) holds")
    return "; ".join(lines)


def _plot(report: dict, out_base: str) -> None:
    """Render the four IoU families to a single PNG (matplotlib optional)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PIVOT step-to-step IoU (GLM-5.2-w4a8 BF16 non-C8, A3)", fontsize=14)
    panels = [
        ("iou_coarse", "coarse set (4096) adjacent-step IoU", axes[0, 0]),
        ("iou_refine", "refine set (2048) adjacent-step IoU", axes[0, 1]),
        ("iou_group", "MTP group-internal IoU (rows i vs i+1)", axes[1, 0]),
    ]
    for key, title, ax in panels:
        st = report[key]
        if not st.get("n"):
            ax.set_title(title + "\n(no data)")
            continue
        ax.hist([0.0, 1.0], weights=[0, 0], bins=40)  # force axis [0,1]
        ax.hist(st.get("raw", []), bins=40, alpha=0.7, color="#4c72b0")
        for stat, color in (("median", "red"), ("p10", "gray"), ("p90", "gray")):
            v = st.get(stat)
            if v is not None:
                ax.axvline(v, color=color, ls="--", lw=1.2,
                           label=f"{stat}={v:.3f}")
        ax.axvline(0.7, color="green", ls=":", lw=1)
        ax.set_xlim(0, 1)
        ax.set_title(f"{title} (n={st['n']}, mean={st.get('mean', float('nan')):.3f})")
        ax.set_xlabel("IoU"); ax.legend(fontsize=8)

    # step-distance decay panel
    ax = axes[1, 1]
    dist = report.get("iou_vs_step_distance", {})
    if dist:
        ds = sorted(int(k) for k in dist)
        med = [dist[str(d)]["median"] for d in ds]
        p10 = [dist[str(d)]["p10"] for d in ds]
        p90 = [dist[str(d)]["p90"] for d in ds]
        ax.plot(ds, med, "o-", color="#4c72b0", label="median")
        ax.fill_between(ds, p10, p90, alpha=0.25, color="#4c72b0", label="p10..p90")
        ax.set_xlabel("step distance d")
        ax.set_title("refine IoU vs step distance (refresh-period guide)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    else:
        ax.set_title("step-distance decay\n(no d>1 chains)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
    print(f"loaded {len(rows)} captures")
    if not rows:
        raise SystemExit("nothing to analyze")

    coarse_all: list[float] = []
    refine_all: list[float] = []
    group_all: list[float] = []
    dist: dict[int, list[float]] = {}
    per_layer: dict[str, dict] = {}

    for p in _pairwise_adjacent(rows):
        layer = p["layer"]
        pl = per_layer.setdefault(layer, {"coarse": [], "refine": []})
        for a, b in p["coarse"]:
            v = _iou(a, b)
            coarse_all.append(v)
            pl["coarse"].append(v)
        for a, b in p["refine"]:
            v = _iou(a, b)
            refine_all.append(v)
            pl["refine"].append(v)
            dist.setdefault(1, []).append(v)

    # Group-internal consistency: for every captured (layer, step, request),
    # compute IoU between consecutive rows i, i+1 of the same request's
    # refine block (g rows per request, rows are laid out request-major:
    # request r occupies rows [r*g, (r+1)*g)). Pure within-step, so it needs
    # no cross-step alignment -- read straight off the raw captures.
    for r in rows:
        g = r["g"]
        if g < 2:
            continue
        ref = r["refine"]  # [D, 2048]
        n_req = ref.shape[0] // g
        for req in range(n_req):
            blk = ref[req * g:(req + 1) * g]
            if blk.shape[0] < 2:
                continue
            for i in range(blk.shape[0] - 1):
                group_all.append(_iou(blk[i], blk[i + 1]))

    # ---- step distance d = 2..5 via cum_start chains (cheap re-walk) ----
    def _chain_pairs(d: int) -> list[tuple]:
        out: list[tuple] = []
        by_layer2: dict[str, list[dict]] = {}
        for r in rows:
            by_layer2.setdefault(r["layer"], []).append(r)
        for layer, lrows in by_layer2.items():
            lrows.sort(key=lambda r: r["step"])
            for i in range(len(lrows) - d):
                a, b = lrows[i], lrows[i + d]
                if a["step"] + d != b["step"] or a["g"] != b["g"]:
                    continue
                g = a["g"]
                ai = {int(s): j for j, s in enumerate(a["cum_start"])}
                bi = {int(s): j for j, s in enumerate(b["cum_start"])}
                for s, ja in ai.items():
                    jb = bi.get(s + d * g)
                    if jb is None:
                        continue
                    ra = a["refine"][ja * g:(ja + 1) * g]
                    rb = b["refine"][jb * g:(jb + 1) * g]
                    if ra.shape[0] == g and rb.shape[0] == g:
                        for k in range(g):
                            out.append((ra[k], rb[k]))
        return out

    for d in range(2, 6):
        vals = [_iou(a, b) for a, b in _chain_pairs(d)]
        if vals:
            dist[d] = vals

    # Raw IoU samples ride along for the histogram panels (kept out of the
    # on-disk JSON to avoid bloat; _plot reads them before the report is
    # serialized).
    report = {
        "iou_coarse": _stats(coarse_all, with_raw=True),
        "iou_refine": _stats(refine_all, with_raw=True),
        "iou_group": _stats(group_all, with_raw=True),
        "iou_vs_step_distance": {str(d): _stats(v) for d, v in sorted(dist.items())},
        "per_layer": {
            k: {"iou_coarse": _stats(v["coarse"]), "iou_refine": _stats(v["refine"])}
            for k, v in per_layer.items()
        },
        "verdict": _verdict(_stats(coarse_all), _stats(refine_all),
                            _stats(group_all)),
    }
    out_base = None
    if args.plot:
        out_base = args.plot[:-4] if args.plot.endswith(".png") else args.plot
    elif args.out:
        out_base = args.out[:-5] if args.out.endswith(".json") else args.out
    if out_base:
        _plot(report, out_base)

    # Strip raw samples before writing the JSON report.
    for key in ("iou_coarse", "iou_refine", "iou_group"):
        report[key].pop("raw", None)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
