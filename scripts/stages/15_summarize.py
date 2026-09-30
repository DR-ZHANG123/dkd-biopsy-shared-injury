"""Stage 15 汇总：dev_metrics.tsv → dev_summary.tsv（均值与跨 cell 范围）与 ablation_summary.tsv
（RRG-ID 相对每个消融 / 对照的 cell 级配对差：均值、bootstrap 95% CI、胜率、Wilcoxon p）。

子集：ALL = 全部 cell；ERCB = 测试单元为 ERCB_*（最接近 DKD 设定的跨批次小样本）；
IGAN_ANALOG = E1 GLOM 测试 ERCB_GLOM_H1 的 IgAN（训练中 IgAN 只来自 IgAN–对照队列，与 DKD 情形同构）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config  # noqa: E402

CFG = load_config()
MET = ["auroc", "adj_auroc", "auroc_ws"]


def _subsets(m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ia = CFG["model2"]["eval"]["igan_analog"]
    return {"ALL": m, "ERCB": m[m.test_unit.str.startswith("ERCB_")],
            "IGAN_ANALOG": m[(m.family == "E1") & (m.compartment == ia["compartment"])
                             & (m.test_unit == ia["test_unit"]) & (m.disease == ia["disease"])]}


def summary_table(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sub, d in _subsets(m).items():
        for keys, g in d.groupby(["family", "method"]):
            for comp in ["ALL", "GLOM", "TUB"]:
                gg = g if comp == "ALL" else g[g.compartment == comp]
                if gg.empty:
                    continue
                r = {"subset": sub, "family": keys[0], "compartment": comp, "method": keys[1], "n_cells": len(gg)}
                for k in MET:
                    v = gg[k].dropna()
                    r.update({f"{k}_mean": v.mean(), f"{k}_min": v.min(), f"{k}_max": v.max()})
                rows.append(r)
    return pd.DataFrame(rows)


def ablation_table(m: pd.DataFrame, ref: str = "RRG-ID") -> pd.DataFrame:
    """配对比较：d = ref − 其他方法（cell 级配对）；均值、cell 级 bootstrap 95% CI、胜率（d > 0 的比例）、Wilcoxon p。
    分组：各评估族、E1+E2、ALL，以及 ERCB 子集。"""
    key = ["family", "compartment", "test_unit", "disease"]
    base = m[m.method == ref].set_index(key)
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    for meth in sorted(set(m.method) - {ref}):
        j = base.join(m[m.method == meth].set_index(key), lsuffix="_ref", rsuffix="_alt", how="inner")
        fam = j.index.get_level_values(0)
        unit = j.index.get_level_values(2)
        groups = {f: fam == f for f in sorted(set(fam))}
        groups.update({"E1+E2": np.isin(fam, ["E1", "E2"]), "ALL": np.ones(len(j), bool),
                       "ERCB": pd.Index(unit).str.startswith("ERCB_")})
        for g, mask in groups.items():
            jj = j[np.asarray(mask)]
            r = {"method": meth, "family": g, "n_cells": len(jj)}
            for k in MET:
                dlt = (jj[f"{k}_ref"] - jj[f"{k}_alt"]).dropna().to_numpy()
                boot = [rng.choice(dlt, len(dlt)).mean() for _ in range(CFG["eval"]["n_boot"])] if len(dlt) else [np.nan]
                r[f"{k}_ref"], r[f"{k}_alt"] = jj[f"{k}_ref"].mean(), jj[f"{k}_alt"].mean()
                r[f"d_{k}_mean"] = dlt.mean() if len(dlt) else np.nan
                r[f"d_{k}_lo"], r[f"d_{k}_hi"] = np.nanquantile(boot, 0.025), np.nanquantile(boot, 0.975)
                r[f"d_{k}_frac_ref_better"] = (dlt > 0).mean() if len(dlt) else np.nan
                r[f"d_{k}_wilcoxon_p"] = wilcoxon(dlt).pvalue if len(dlt) >= 6 and (dlt != 0).any() else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def summarize(out: Path) -> None:
    m = pd.read_csv(out / "dev_metrics.tsv", sep="\t")
    s = summary_table(m)
    s.to_csv(out / "dev_summary.tsv", sep="\t", index=False)
    a = ablation_table(m) if "RRG-ID" in set(m.method) else pd.DataFrame()
    a.to_csv(out / "ablation_summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    show = s[s.compartment == "ALL"][["subset", "family", "method", "n_cells", "auroc_mean", "auroc_min",
                                       "auroc_max", "adj_auroc_mean", "auroc_ws_mean"]]
    print(show.round(3).to_string(index=False))
    if not a.empty:
        print(a[a.family == "ALL"][["method", "n_cells", "d_auroc_mean", "d_auroc_lo", "d_auroc_hi",
                                    "d_auroc_frac_ref_better", "d_adj_auroc_mean", "d_adj_auroc_lo", "d_adj_auroc_hi",
                                    "d_adj_auroc_frac_ref_better"]].round(3).to_string(index=False))


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "dev"
    summarize(ROOT / "results" / "15_model" / tag)
