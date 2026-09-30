"""Stage 22 S1：adaptive / failed-repair 占比与 eGFR 的（偏）相关 —— 排除 AKI 供体前后并列。

输入 = stage 20d 供体表（results/20_repair_state/donors/donor_table.tsv；KPMP snRNA，eGFR = 分箱中点，
SCP 分数 = 组织 pseudobulk 的 injury–repair score）。与 stage 20d 同一方法：Spearman（置换 P）与控制 injury–repair
score 的偏 Spearman（两者对分数的秩残差），另加供体 bootstrap 95% 区间。frPT 与 frTAL 分开。
供体子集见 config sensitivity22.s1.subsets（全部已分类、去掉 AKI、REF+CKD、仅 CKD、仅 AKI）。
产出 results/22_sensitivity/S1_egfr/{partial_egfr.tsv, subset_egfr.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repair_state import perm_spearman  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
R3 = CFG["sensitivity22"]
P = R3["s1"]
SRC = ROOT / "results/20_repair_state/donors/donor_table.tsv"
O = ROOT / "results/22_sensitivity/S1_egfr"


def rank_resid(v: np.ndarray, c: np.ndarray) -> np.ndarray:
    rv, rc = stats.rankdata(v), stats.rankdata(c)
    X = np.c_[np.ones(len(rc)), rc]
    return rv - X @ np.linalg.lstsq(X, rv, rcond=None)[0]


def partial_rho(x, y, c) -> float:
    return float(np.corrcoef(stats.rankdata(rank_resid(x, c)), stats.rankdata(rank_resid(y, c)))[0, 1])


def one(t: pd.DataFrame, col: str, rng) -> dict:
    x, y, c = t[col].to_numpy(float), t.egfr_mid.to_numpy(float), t.scp_full.to_numpy(float)
    r, p = perm_spearman(x, y, R3["n_perm"], rng)
    rx, ry = rank_resid(x, c), rank_resid(y, c)
    pr, pp = perm_spearman(rx, ry, R3["n_perm"], rng)
    boot = []
    for _ in range(R3["n_boot"]):
        j = rng.integers(0, len(t), len(t))
        if np.unique(y[j]).size < 3:
            continue
        boot.append(partial_rho(x[j], y[j], c[j]))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n": len(t), "spearman": r, "p_perm": p, "partial_spearman_given_score": pr, "p_perm_partial": pp,
            "partial_lo": float(lo), "partial_hi": float(hi)}


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    W = pd.read_csv(SRC, sep="\t", dtype={"donor": str})
    W = W[W.dataset == P["dataset"]]
    rows, desc = [], []
    for sub, cats in P["subsets"].items():
        w = W[W.category.isin(cats)]
        desc.append({"subset": sub, "categories": ",".join(cats), "n_donors": len(w),
                     "n_with_egfr": int(w.egfr_mid.notna().sum()),
                     "median_egfr": float(w.egfr_mid.median()),
                     "n_AKI": int((w.category == "AKI").sum())})
        for col in P["states"]:
            t = w[w[col].notna() & w.egfr_mid.notna() & w.scp_full.notna()]
            if len(t) < P["min_donors"]:
                continue
            rows.append({"subset": sub, "state": col, **one(t, col, rng)})
    R = pd.DataFrame(rows)
    D = pd.DataFrame(desc)
    R.to_csv(O / "partial_egfr.tsv", sep="\t", index=False)
    D.to_csv(O / "subset_egfr.tsv", sep="\t", index=False)
    write_provenance("22_sensitivity/S1_egfr", [SRC], [O / "partial_egfr.tsv", O / "subset_egfr.tsv"], CFG["seed"],
                     {"s1": P, "n_perm": R3["n_perm"], "n_boot": R3["n_boot"]})
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(D.to_string())
        print(R.round(3).to_string())


if __name__ == "__main__":
    main()
