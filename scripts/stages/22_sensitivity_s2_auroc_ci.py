"""Stage 22 S2：独立队列中 injury–repair score 疾病 vs 对照 AUROC 的患者 bootstrap 95% 区间。

输入 = stage 19d 的样本分数（results/19_shared_program/replicate/scores.tsv，gene_set = full，区室 = 该队列的主核心），
与 stage 19d 相同的阳性诊断定义（config shared_program.replicate.cohorts）。每个患者（KPMP 为 participant）一个样本；
bootstrap 在病人组与对照组内分别有放回抽取患者（分层，n_boot 次），报告百分位区间与 AUROC ≤ 0.5 的 bootstrap 比例。
对照：DKD vs 对照（有 DKD 的队列）。stage 19d 已有的区间（auroc_lo/auroc_hi）一并列出以便核对。
产出 results/22_sensitivity/S2_auroc_ci/{auroc_ci.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from lib.stats import auc_rows  # noqa: E402

CFG = load_config()
R3 = CFG["sensitivity22"]
RP = CFG["shared_program"]["replicate"]
SRC = ROOT / "results/19_shared_program/replicate/scores.tsv"
S19 = ROOT / "results/19_shared_program/replicate/auroc.tsv"
TS = ROOT / "results/17_independent/test_samples.tsv"
O = ROOT / "results/22_sensitivity/S2_auroc_ci"


def boot(y: np.ndarray, x: np.ndarray, cl: np.ndarray, n: int, rng) -> np.ndarray:
    groups = []
    for lab in (1, 0):
        c = np.unique(cl[y == lab])
        groups.append([np.flatnonzero((cl == k) & (y == lab)) for k in c])
    out = np.empty(n)
    for b in range(n):
        j = np.concatenate([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]) for g in groups])
        out[b] = auc_rows(y[j], x[j][None])[0]
    return out


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(SRC, sep="\t")
    part = pd.read_csv(TS, sep="\t", index_col=0, usecols=["sample_uid", "participant"], low_memory=False).participant
    ref = pd.read_csv(S19, sep="\t")
    ref = ref[(ref.gene_set == R3["s2"]["gene_set"]) & ref.primary]
    rows = []
    for name, spec in RP["cohorts"].items():
        s = S[(S.cohort == name) & (S.core == spec["core"])].copy()
        s["cluster"] = s.sample_uid.map(part).fillna(s.sample_uid)
        contrasts = {"patients_vs_control": spec["positive"]}
        if "DKD" in set(s.diagnosis) and spec["positive"] != ["DKD"]:
            contrasts["DKD_vs_control"] = ["DKD"]
        for cname, pos in contrasts.items():
            t = s[s.diagnosis.isin(pos + ["CONTROL"])]
            y = t.diagnosis.isin(pos).to_numpy(int)
            x = t.scp.to_numpy(float)
            auc = float(auc_rows(y, x[None])[0])
            b = boot(y, x, t.cluster.astype(str).to_numpy(), R3["n_boot"], rng)
            lo, hi = np.quantile(b, [0.025, 0.975])
            r = ref[ref.cohort == name]
            rows.append({"cohort": name, "compartment_core": spec["core"], "contrast": cname, "positive": "+".join(pos),
                         "n_pos": int(y.sum()), "n_ctrl": int((1 - y).sum()), "n_clusters": t.cluster.nunique(),
                         "auroc": auc, "ci_lo": float(lo), "ci_hi": float(hi), "boot_frac_le_0.5": float((b <= 0.5).mean()),
                         "stage19_auroc": float(r.auroc.iloc[0]) if cname == "patients_vs_control" and len(r) else np.nan,
                         "stage19_lo": float(r.auroc_lo.iloc[0]) if cname == "patients_vs_control" and len(r) else np.nan,
                         "stage19_hi": float(r.auroc_hi.iloc[0]) if cname == "patients_vs_control" and len(r) else np.nan,
                         "pct_vs_random": float(r.pct_vs_random.iloc[0]) if cname == "patients_vs_control" and len(r) else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(O / "auroc_ci.tsv", sep="\t", index=False)
    write_provenance("22_sensitivity/S2_auroc_ci", [SRC, S19, TS], [O / "auroc_ci.tsv"], CFG["seed"],
                     {"n_boot": R3["n_boot"], "gene_set": R3["s2"]["gene_set"]})
    with pd.option_context("display.width", 250):
        print(R.round(3).to_string())


if __name__ == "__main__":
    main()
