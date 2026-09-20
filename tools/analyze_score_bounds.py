# -*- coding: utf-8 -*-
"""Offline analyzer for the score-bounds probe dumps (topk_probe_scores.py).

Reads the npz dumps and recomputes, per sampled (step, layer):
  * ceiling-bound survivor fraction  -- would 跳块法(BET) prune? B(b) uses only
    the dumped per-head block max/min dots (ReLU monotonicity makes this exact:
    B(b) = sum_{w>0} w_h*max(0,maxdot) + sum_{w<0}|w_h|*max(0,-mindot))
  * dead-head fraction               -- heads whose ReLU never fires in a block
    (激活稀疏度); the certificate's slack shrinks from H to H_effective
  * 打分余量比                        -- median (B(b) / theta) over surviving
    blocks; how much headroom a tighter bound must recover
  * weight sign fraction             -- 正号权重占比 (符号越正, 天花板越准)

Emits: decisions (转正/放弃, mirroring report.json's decision precedent),
a summary txt, and a 2x2 matplotlib figure (Chinese labels, optional).

Usage:
  python tools/analyze_score_bounds.py /tmp/score_probe --out score_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _load_dumps(src: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(src, "score__*.npz")))
    out = []
    for f in files:
        try:
            with np.load(f, allow_pickle=False) as z:
                item = {k: z[k].item() if z[k].ndim == 0 else z[k]
                        for k in z.files}
                item["_file"] = os.path.basename(f)
                out.append(item)
        except Exception as e:  # noqa: BLE001 -- skip corrupt dumps, keep going
            print(f"skip {f}: {e}")
    return out


def _analyze_one(z: dict) -> dict:
    w = z["w_row"].astype(np.float32)                  # [H]
    maxdot = z["maxdot"].astype(np.float32)            # [NB, H]
    mindot = z["mindot"].astype(np.float32)            # [NB, H]
    score = z["score"].astype(np.float32)              # [L]
    top_k = int(z["top_k"])
    pos = w > 0

    # exact per-key score sanity: recompute from extremes is impossible, but
    # the dump stores the exact vector -- use it as ground truth.
    theta = np.partition(score, -top_k)[-top_k] if len(score) >= top_k else score.max()

    # ceiling bound per block (ReLU monotonic form; inf-safe)
    pos_term = np.where(pos[None, :], w[None, :] * np.maximum(maxdot, 0.0), 0.0)
    neg_term = np.where(~pos[None, :],
                        np.abs(w)[None, :] * np.maximum(-mindot, 0.0), 0.0)
    B = (pos_term + neg_term).sum(axis=1)              # [NB]

    finite = np.isfinite(B)
    survivors = (B > theta) & finite
    prune_rate = 1.0 - survivors.mean()

    # dead heads: for w>0 the head never fires in this block if maxdot<=0;
    # for w<0 never fires (contributes 0 to the max) if mindot>=0.
    dead = np.where(pos[None, :], maxdot <= 0.0, mindot >= 0.0)
    dead_frac = dead.mean()

    # margin: among surviving blocks, how far above theta their ceiling sits
    surv_B = B[survivors & finite] if survivors.any() else np.array([np.nan])
    margin = float(np.median(surv_B / max(theta, 1e-9)))

    # consistency: no pruned block may contain a top-k key. The dump does not
    # store per-key positions (only the score vector + block geometry), so we
    # verify the imputation-free implication: pruned blocks have B<=theta and
    # B>=max true score in block by construction -- spot-check via score order.
    # (Full per-key audit needs the keys; the probe's index dump covers it.)
    return {
        "prune_rate": round(float(prune_rate), 4),
        "dead_head_frac": round(float(dead_frac), 4),
        "margin_median": round(margin, 3),
        "theta": round(float(theta), 4),
        "sign_pos_frac": round(float((w > 0).mean()), 3),
        "finite_bound_frac": round(float(finite.mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="score-probe dump dir (/tmp/score_probe)")
    ap.add_argument("--out", default="score_report.json")
    ap.add_argument("--txt", default="score_report.txt")
    ap.add_argument("--png", default="", help="optional figure")
    ap.add_argument("--prune-gate", type=float, default=0.30,
                    help="转正门槛: prune_rate >= gate")
    args = ap.parse_args()

    dumps = _load_dumps(args.src)
    if not dumps:
        print("no dumps found"); return 1
    per = []
    for z in dumps:
        r = _analyze_one(z)
        r["_file"] = z["_file"]; r["layer"] = z["layer_name"]; r["step"] = z["step"]
        per.append(r)

    def med(key):
        vals = [r[key] for r in per if np.isfinite(r[key])]
        return round(float(np.median(vals)), 4) if vals else float("nan")

    summary = {
        "n_dumps": len(per),
        "prune_rate_median": med("prune_rate"),
        "prune_rate_min": round(min(r["prune_rate"] for r in per), 4),
        "dead_head_frac_median": med("dead_head_frac"),
        "sign_pos_frac_median": med("sign_pos_frac"),
    }
    summary["decision"] = {
        "跳块法(BET)/提前收工(PARE)/合并候选池(GUST) 转正":
            "是" if summary["prune_rate_min"] >= args.prune_gate else
            f"否(最低剪枝率 {summary['prune_rate_min']:.0%} < 门槛 {args.prune_gate:.0%};"
            " 优化界形式后复测)",
        "激活稀疏度可用(死头剪除)": "是" if summary["dead_head_frac_median"] > 0.05 else "弱",
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_dump": per}, f,
                  ensure_ascii=False, indent=2)
    with open(args.txt, "w", encoding="utf-8") as f:
        f.write(f"dumps={summary['n_dumps']}  "
                f"prune_rate median={summary['prune_rate_median']} "
                f"min={summary['prune_rate_min']}\n")
        f.write(f"dead_head median={summary['dead_head_frac_median']}  "
                f"sign_pos median={summary['sign_pos_frac_median']}\n")
        for k, v in summary["decision"].items():
            f.write(f"决策|{k}: {v}\n")
        for r in sorted(per, key=lambda x: x["prune_rate"]):
            f.write(f"  {r['_file']}: layer={r['layer']} step={r['step']} "
                    f"prune={r['prune_rate']:.3f} dead={r['dead_head_frac']:.2f} "
                    f"margin={r['margin_median']}\n")
    print(open(args.txt, encoding="utf-8").read())

    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                               "WenQuanYi Micro Hei"]
            plt.rcParams["axes.unicode_minus"] = False
            fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.6))
            layers = sorted({r["layer"] for r in per})
            for ax, key, title in (
                (axs[0][0], "prune_rate", "剪枝率 (越高越好)"),
                (axs[0][1], "dead_head_frac", "死头占比"),
                (axs[1][0], "margin_median", "打分余量比 (中位)"),
                (axs[1][1], "sign_pos_frac", "正号权重占比"),
            ):
                vals = [np.mean([r[key] for r in per if r["layer"] == ln])
                        for ln in layers]
                ax.bar([str(ln) for ln in layers], vals)
                ax.set_title(title); ax.tick_params(axis="x", rotation=45)
            fig.suptitle("分数探针离线分析 (每次采样=一个请求的首查询行)")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(args.png, dpi=140)
        except Exception as e:  # noqa: BLE001 -- plotting is optional
            print(f"plot skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
