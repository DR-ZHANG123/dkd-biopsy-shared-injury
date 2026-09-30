"""Stage 20d：KPMP 供体水平 —— 修复失败 / 退行 / 增殖状态占 PT、TAL 的比例：各病理类别 vs 健康参考、与 SCP 分数和 eGFR 的关系。

比例来自 stage 20a fractions（供体 × 谱系细胞计数；谱系细胞 < min_cells 的供体不计）。
类别检验：logit(比例) 的 A − B 均值差、Hedges g、AUROC、双侧供体标签置换 P；snRNA 另做取材匹配（经皮活检 CKD vs 经皮活检 REF）。
SCP 分数 = stage 19c 组织 pseudobulk（TUB 核心，full 与 state_only 场景）。eGFR = 分箱中点（仅 snRNA 有）。
Spearman 的 P = 置换；偏 Spearman（控制 SCP，秩残差）检验状态占比是否携带 SCP 之外的 eGFR 信息。
snRNA 为主：scRNA 需要组织解离，应激 / 损伤状态占比受解离偏倚影响，只作方向复现。
产出 results/20_repair_state/donors/
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repair_state import CFG, OUT, RS, group_contrast, logit, perm_spearman  # noqa: E402
from lib.repro import set_global_seed, write_provenance  # noqa: E402
from lib.scp_core import OUT as SCP_OUT  # noqa: E402

O = OUT / "donors"
PERC = CFG["shared_program"]["kpmp"]["percutaneous"]


def egfr_mid(v) -> float:
    m = re.match(r"(\d+)-(\d+)", str(v))
    return (int(m.group(1)) + int(m.group(2)) + 1) / 2 if m else np.nan


def wide_table(fr: pd.DataFrame) -> pd.DataFrame:
    f = fr[fr.n_lineage >= RS["min_cells"]].copy()
    f["key"] = f.lineage + ":" + f.state
    W = f.pivot_table(index=["dataset", "donor"], columns="key", values="frac").reset_index()
    base = fr.drop_duplicates(["dataset", "donor"])[["dataset", "donor", "category", "procurement", "egfr_bin"]]
    W = base.merge(W, on=["dataset", "donor"], how="left")
    W["egfr_mid"] = W.egfr_bin.map(egfr_mid)
    s19 = pd.read_csv(SCP_OUT / "kpmp" / "decomp_scores.tsv.gz", sep="\t", dtype={"donor": str})
    s19 = s19[(s19.compartment == RS["compartment"]) & s19.scenario.isin(["full", "state_only", "comp_only"])]
    s = s19.pivot_table(index=["dataset", "donor"], columns="scenario", values="score").add_prefix("scp_").reset_index()
    return W.merge(s, on=["dataset", "donor"], how="left")


def category_tests(W: pd.DataFrame, cols: list[str], rng) -> pd.DataFrame:
    rows = []
    for ds, w in W.groupby("dataset"):
        w = w.set_index("donor")
        cat = w.category
        groups = {g: cat.index[cat == g] for g in ("DKD", "HKD", "OTHER", "CKD_unadj", "AKI", "DM_R")}
        groups["CKD"] = cat.index[cat.isin(RS["ckd"])]
        ref = cat.index[cat == "REF"]
        perc = w.index[w.procurement == PERC]
        pairs = [(g, "REF", idx, ref) for g, idx in groups.items()]
        if len(ref.intersection(perc)) >= RS["min_donors"]:
            pairs.append(("CKD_perc", "REF_perc", groups["CKD"].intersection(perc), ref.intersection(perc)))
        for c in cols:
            v = w[c].to_numpy(float)
            lv = logit(v, RS["bulk"]["eps"])
            lv[~np.isfinite(v)] = np.nan
            for a, b, ia, ib in pairs:
                r = group_contrast(lv, pd.Index(w.index), ia, ib, RS["n_perm"], rng)
                if r.get("n_A", 0) < 3 or r.get("n_B", 0) < 3:
                    continue
                va = w.loc[ia, c].dropna()
                vb = w.loc[ib, c].dropna()
                rows.append({"dataset": ds, "state": c, "group_A": a, "group_B": b,
                             "median_frac_A": va.median(), "median_frac_B": vb.median(),
                             "mean_frac_A": va.mean(), "mean_frac_B": vb.mean(),
                             "descriptive_only": r["n_A"] < RS["min_donors"], **r})
    return pd.DataFrame(rows)


def rank_resid(v: np.ndarray, c: np.ndarray) -> np.ndarray:
    rv, rc = stats.rankdata(v), stats.rankdata(c)
    X = np.c_[np.ones(len(rc)), rc]
    return rv - X @ np.linalg.lstsq(X, rv, rcond=None)[0]


def correlations(W: pd.DataFrame, cols: list[str], rng) -> pd.DataFrame:
    rows = []
    subsets = {"CKD": lambda w: w.category.isin(RS["ckd"]), "REF+CKD": lambda w: w.category.isin(["REF"] + RS["ckd"]),
               "CKD+AKI+DM_R+CKD_unadj": lambda w: w.category.isin(RS["ckd"] + ["AKI", "DM_R", "CKD_unadj"]),
               "all_categorised": lambda w: w.category.notna()}
    for ds, w in W.groupby("dataset"):
        for sub, fn in subsets.items():
            m = fn(w)
            for c in cols + ["scp_full", "scp_state_only"]:
                for target in ("scp_full", "scp_state_only", "egfr_mid"):
                    if c == target or (c.startswith("scp_") and target.startswith("scp_")):
                        continue
                    t = w[m & w[c].notna() & w[target].notna()]
                    if len(t) < 10:
                        continue
                    r, p = perm_spearman(t[c].to_numpy(float), t[target].to_numpy(float), RS["n_perm"], rng)
                    row = {"dataset": ds, "donors": sub, "variable": c, "target": target, "n": len(t), "spearman": r,
                           "p_perm": p}
                    if target == "egfr_mid" and not c.startswith("scp_"):
                        tt = t[t.scp_full.notna()]
                        rx = rank_resid(tt[c].to_numpy(float), tt.scp_full.to_numpy(float))
                        ry = rank_resid(tt.egfr_mid.to_numpy(float), tt.scp_full.to_numpy(float))
                        pr, pp = perm_spearman(rx, ry, RS["n_perm"], rng)
                        row |= {"partial_spearman_given_scp": pr, "p_perm_partial": pp}
                    rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    fr = pd.read_csv(OUT / "programs" / "fractions.tsv.gz", sep="\t", dtype={"donor": str})
    W = wide_table(fr)
    cols = [c for c in W.columns if ":" in c and not c.endswith(":normal")]
    A = category_tests(W, cols, rng)
    R = correlations(W, cols, rng)
    outs = {"donor_table.tsv": W, "category_tests.tsv": A, "correlations.tsv": R}
    for f, t in outs.items():
        t.to_csv(O / f, sep="\t", index=False)
    write_provenance("20_repair_state/donors", [OUT / "programs/fractions.tsv.gz", SCP_OUT / "kpmp/decomp_scores.tsv.gz"],
                     [O / f for f in outs], CFG["seed"], {"n_perm": RS["n_perm"], "min_cells": RS["min_cells"]})
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        key = ["PT:rfPT", "PT:aPT", "PT:frPT", "PT:dPT", "PT:cycPT", "TAL:rfTAL", "TAL:aTAL", "TAL:frTAL", "TAL:dTAL"]
        print(A[A.state.isin(key)][["dataset", "state", "group_A", "group_B", "n_A", "n_B", "median_frac_A", "median_frac_B",
                                    "hedges_g", "auroc", "p_perm"]].round(3).to_string())
        print(R[R.variable.isin(key + ["scp_full", "scp_state_only"])].round(3).to_string())


if __name__ == "__main__":
    main()
