# -*- coding: utf-8 -*-
"""ctx_sweep CSV → 论文图:横轴上下文挡位,coarse/refine/LI 三柱,4 个批次挡位子图。

输入: bench_two_ops_ctx_sweep.py --csv ctx_sweep.csv 的输出(含 li_us 列)。
布局: 2×2 子图,每子图一个 req 挡位;柱 = coarse_us / refine_us / li_us(原生基线);
status 列非 ok 的配置自动剔除(缺失挡位红字标注)。
下游复现: python plot_ctx_sweep.py ctx_sweep.csv out.png
  --pools 可叠加第二组 CSV(pool_sweep)时暂不支持;单图只吃 ctx_sweep.csv。
"""
import argparse
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REQ_ORDER = (4, 8, 16, 32)
CTX_ORDER = (2, 10, 20, 64, 128, 256)
SERIES = (("coarse_screen", "coarse_us", "#4C72B0"),
          ("refine", "refine_us", "#DD8452"),
          ("native LI (baseline)", "li_us", "#55A868"))


def main(csv_path, out_path):
    data = defaultdict(dict)   # req -> ctx_k -> {col: us}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"].strip() != "ok":
                continue
            req, ctx = int(r["req"]), int(r["ctx_k"])
            cell = {}
            for _, col, _ in SERIES:
                if r.get(col):
                    cell[col] = float(r[col])
            if cell:
                data[req][ctx] = cell

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()
    n = len(SERIES)
    width = 0.8 / n
    for ax, req in zip(axes, REQ_ORDER):
        d = data.get(req, {})
        xs = [c for c in CTX_ORDER if c in d]
        x = np.arange(len(xs))
        for i, (label, col, color) in enumerate(SERIES):
            vals = [d[c].get(col, np.nan) for c in xs]
            ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
                   label=label, color=color)
            for xi, v in zip(x, vals):
                if v == v:
                    ax.annotate(f"{v:.0f}", (xi + (i - (n - 1) / 2) * width, v),
                                ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}K" for c in xs])
        ax.set_title(f"req = {req}  (refine/LI rows = {req * 4})")
        ax.set_ylabel("kernel time (µs)")
        ax.grid(axis="y", alpha=0.3)
        missing = [c for c in CTX_ORDER if c not in d]
        if missing:
            ax.text(0.02, 0.96, "missing: " + ",".join(f"{c}K" for c in missing),
                    transform=ax.transAxes, fontsize=7, color="red", va="top")
    axes[0].legend()
    fig.suptitle("PIVOT chain (coarse+refine) vs native LightningIndexer "
                 "kernel time vs context length (pool_w = 4103 prod)")
    fig.supxlabel("context length per row")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out")
    a = ap.parse_args()
    main(a.csv, a.out)
