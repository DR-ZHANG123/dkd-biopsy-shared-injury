"""Stage 11b：共享疾病–对照轴的三项严格检验。

1. 拆半对照：同一单元内，DKD 偏移与其他 CKD 偏移各用互不重叠的一半对照计算，
   与「其他 CKD 自身拆半」的信度比较（重复 n_split 次）。
2. 跨队列特异性比：r(DKD_A, 其他CKD_B) / r(DKD_A, DKD_B)，A、B 为不共享标本的单元。
3. 外部标尺：GSE175759 的 eGFR（剔除作者标注的技术离群），损伤轴只用其余肾小管来源估计。
产出 results/11b_shared_axis/{split_controls,specificity_ratio,egfr_check}.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.injury import NON_DISEASE, full_ranks, injury_axis, injury_score  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
SA = CFG["shared_axis_tests"]
OUT = ROOT / "results" / "11b_shared_axis"


def groups(idx, s):
    d = s.loc[idx, "diagnosis"].values
    return d == "CONTROL", d == "DKD", ~np.isin(d, list(NON_DISEASE))


def split_controls(units, s, rng) -> pd.DataFrame:
    rows = []
    for u in SA["split_units"]:
        idx = units[u]
        R = full_ranks(idx, s).dropna(axis=1).values
        ctrl, dkd, oth = groups(idx, s)
        ci, oi = np.where(ctrl)[0], np.where(oth)[0]
        for rep in range(SA["n_split"]):
            p = rng.permutation(ci)
            A, B = p[: len(p) // 2], p[len(p) // 2:]
            q = rng.permutation(oi)
            O1, O2 = q[: len(q) // 2], q[len(q) // 2:]
            mA, mB = R[A].mean(0), R[B].mean(0)
            rows.append({"unit": u, "rep": rep,
                         "r_shared_controls": spearmanr(R[dkd].mean(0) - R[ci].mean(0), R[oth].mean(0) - R[ci].mean(0))[0],
                         "r_split_dkd_other": spearmanr(R[dkd].mean(0) - mA, R[oth].mean(0) - mB)[0],
                         "r_split_other_reliability": spearmanr(R[O1].mean(0) - mA, R[O2].mean(0) - mB)[0]})
    return pd.DataFrame(rows)


def shift(u, which, units, s):
    idx = units[u]
    R = full_ranks(idx, s).dropna(axis=1)
    ctrl, dkd, oth = groups(idx, s)
    m = dkd if which == "DKD" else oth
    return R[m].mean() - R[ctrl].mean()


def specificity(units, s) -> pd.DataFrame:
    rows = []
    for a, b in SA["ratio_pairs"]:
        da, db, ob = shift(a, "DKD", units, s), shift(b, "DKD", units, s), shift(b, "OTH", units, s)
        g = da.index.intersection(db.index).intersection(ob.index)
        r_dd, r_do = spearmanr(da[g], db[g])[0], spearmanr(da[g], ob[g])[0]
        rows.append({"unit_A_dkd": a, "unit_B": b, "n_genes": len(g), "r_dkdA_dkdB": r_dd,
                     "r_dkdA_otherB": r_do, "specificity_ratio": r_do / r_dd,
                     "informative": bool(r_dd >= SA["min_r_dd"])})
    return pd.DataFrame(rows)


def egfr_check(units, s) -> pd.DataFrame:
    u = SA["egfr_unit"]
    m = pd.read_csv(ROOT / CFG["paths"]["interim"] / f"{u}_meta.tsv", sep="\t", dtype=str)
    ch = m.filter(like="characteristics").fillna("").agg(" ; ".join, axis=1)
    egfr = ch.str.extract(r"estimated gfr[^:]*:\s*([\d.]+)")[0].astype(float)
    outlier = ch.str.contains(r"technical outlier:\s*(?:yes|outlier)", case=False)
    e = pd.Series(egfr.values, index=f"{u}|" + m["gsm"])[~outlier.values].dropna()
    axis = injury_axis([units[x] for x in SA["egfr_axis_sources"]], s)
    idx = e.index.intersection(s.index)
    R = full_ranks(idx, s).dropna(axis=1)
    inj = injury_score(R, axis)
    ctrl = (s.loc[idx, "diagnosis"] == "CONTROL").values
    rows = [{"test": "patients_vs_controls_auroc", "n": len(idx), "value": roc_auc_score(~ctrl, inj), "p": np.nan}]
    for name, mask in (("rho_injury_egfr_all", np.ones(len(idx), bool)), ("rho_injury_egfr_patients", ~ctrl)):
        r = spearmanr(inj[mask], e[idx][mask])
        rows.append({"test": name, "n": int(mask.sum()), "value": r[0], "p": r[1]})
    for g in SA["egfr_marker_genes"]:
        if g in R:
            r = spearmanr(R[g][~ctrl], e[idx][~ctrl])
            rows.append({"test": f"rho_{g}_egfr_patients", "n": int((~ctrl).sum()), "value": r[0], "p": r[1]})
    rows.append({"test": "egfr_median_patients", "n": int((~ctrl).sum()), "value": float(e[idx][~ctrl].median()), "p": np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    from importlib import import_module
    s, units = import_module("11_injury_axis").load_samples()
    rng = np.random.default_rng(CFG["seed"])
    sp = split_controls(units, s, rng)
    sp.to_csv(OUT / "split_controls.tsv", sep="\t", index=False)
    sr = specificity(units, s)
    sr.to_csv(OUT / "specificity_ratio.tsv", sep="\t", index=False)
    eg = egfr_check(units, s)
    eg.to_csv(OUT / "egfr_check.tsv", sep="\t", index=False)
    write_provenance("11b_shared_axis", [ROOT / "data/processed/samples.tsv"],
                     [OUT / f for f in ("split_controls.tsv", "specificity_ratio.tsv", "egfr_check.tsv")], CFG["seed"])
    print(sp.groupby("unit").mean(numeric_only=True).drop(columns="rep").round(3).to_string())
    print(sr.round(3).to_string())
    print(eg.round(3).to_string())


if __name__ == "__main__":
    main()
