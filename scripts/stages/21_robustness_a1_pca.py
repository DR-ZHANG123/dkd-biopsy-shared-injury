"""Stage 21 A1 补充：样本水平上 injury–repair 分数在活检转录组总变异中的位置（「dominates transcriptomes」的直接检验）。

每个评估单元（config injury.compartments.*.eval；全部样本 = 病人 + 对照）：
  样本内秩矩阵中心化后 PCA：前 5 个 PC 的方差份额，各 PC 与 stage 11 外部分数（results/11_injury/injury_scores.tsv）的 |Spearman|；
  分数解释的转录组总方差 = Σ_g var_g·R²_g / Σ_g var_g（每个基因对分数单变量回归）；
  零分布：同大小（top_k 上 + top_k 下）随机基因集分数解释的总方差（n_perm 个），报告中位数与 95% 分位。
产出 results/21_robustness/A1_share/pca/{transcriptome_pca.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.injury import full_ranks  # noqa: E402
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.robustness21 import CFG, OUT, RV, units  # noqa: E402

O = OUT / "A1_share" / "pca"
INJ = CFG["injury"]


def var_explained(Xc: np.ndarray, z: np.ndarray) -> float:
    zc = (z - z.mean()) / (np.linalg.norm(z - z.mean()) + 1e-12)
    proj = zc @ Xc                                       # 每个基因在 z 上的回归平方和的平方根
    return float((proj ** 2).sum() / (Xc ** 2).sum())


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, U = units()
    sc = pd.read_csv(ROOT / "results/11_injury/injury_scores.tsv", sep="\t")
    rows = []
    k = INJ["top_k"]
    for comp, spec in INJ["compartments"].items():
        for u in spec["eval"]:
            R = full_ranks(U[u], s).dropna(axis=1)
            z = sc[sc.eval_unit == u].set_index("sample_uid").injury.reindex(R.index).to_numpy(float)
            X = R.to_numpy(float)
            Xc = X - X.mean(0)
            Uu, S, _ = np.linalg.svd(Xc, full_matrices=False)
            ve = S ** 2 / (S ** 2).sum()
            obs = var_explained(Xc, z)
            null = []
            for _ in range(RV["n_perm"]):
                pick = rng.choice(X.shape[1], 2 * k, replace=False)
                null.append(var_explained(Xc, X[:, pick[:k]].mean(1) - X[:, pick[k:]].mean(1)))
            null = np.array(null)
            ctrl = (s.loc[R.index, "diagnosis"] == "CONTROL").to_numpy()
            row = {"compartment": comp, "unit": u, "n_samples": len(X), "n_controls": int(ctrl.sum()), "n_genes": X.shape[1],
                   "var_explained_by_score": obs, "null_median": float(np.median(null)),
                   "null_q95": float(np.quantile(null, 0.95)), "p": float((1 + (null >= obs).sum()) / (1 + len(null)))}
            for i in range(5):
                row[f"pc{i + 1}_var_frac"] = ve[i]
                row[f"pc{i + 1}_abs_spearman_score"] = abs(spearmanr(Uu[:, i], z)[0])
            rho = [abs(spearmanr(Uu[:, i], z)[0]) for i in range(5)]
            row["pc_best_matching_score"] = int(np.argmax(rho) + 1)
            rows.append(row)
            print(u, "done", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(O / "transcriptome_pca.tsv", sep="\t", index=False)
    write_provenance("21_robustness/A1_share/pca", [ROOT / "results/11_injury/injury_scores.tsv"], [O / "transcriptome_pca.tsv"],
                     CFG["seed"], {"n_perm": RV["n_perm"]})
    with pd.option_context("display.width", 250):
        print(T.round(3).to_string())


if __name__ == "__main__":
    main()
