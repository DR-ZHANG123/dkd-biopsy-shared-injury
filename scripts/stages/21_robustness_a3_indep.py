"""Stage 21 A3（独立队列部分）：反应基因同时剔除取材敏感与经典即早基因（及各自单独剔除）后，独立队列的
疾病 vs 健康 AUROC 与临床关联（样本、阳性病种、临床字段与 stage 19d 完全相同；只报告点估计与置换 P，不重新选择任何参数）。
产出 results/21_robustness/A3_sensitivity/indep/{auroc.tsv, clinical.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.repair_state import perm_spearman  # noqa: E402
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.robustness21 import CFG, OUT, RV, core_exclude, units  # noqa: E402
from lib.scp import signed_score  # noqa: E402
from lib.scp_core import load_core  # noqa: E402
from lib.stats import auc_rows  # noqa: E402

O = OUT / "A3_sensitivity" / "indep"


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, U = units()
    S19 = import_module("19_shared_program_replicate")
    aucs, clin = [], []
    for name, spec in CFG["shared_program"]["replicate"]["cohorts"].items():
        R, meta = S19.load_cohort(name, spec, s, U)
        d = meta.diagnosis.to_numpy()
        m = np.isin(d, spec["positive"] + ["CONTROL"])
        y = np.isin(d[m], spec["positive"]).astype(int)
        up0, dn0 = load_core(spec["core"])
        for var in RV["gene_variants"]:
            ex = core_exclude(var, spec["core"])
            up = [g for g in up0 if g not in ex and g in R.columns]
            dn = [g for g in dn0 if g not in ex and g in R.columns]
            sc = signed_score(R, up, dn).to_numpy()
            aucs.append({"cohort": name, "core": spec["core"], "variant": var, "n_up": len(up), "n_down": len(dn),
                         "n_pos": int(y.sum()), "n_ctrl": int((1 - y).sum()), "auroc": float(auc_rows(y, sc[m][None])[0])})
            for test, mk, x in S19.clinical(name, meta):
                if mk.sum() < 5:
                    continue
                row = {"cohort": name, "variant": var, "test": test, "n": int(mk.sum())}
                if test == "advanced_vs_early_DN":
                    row["auroc"] = float(auc_rows(x[mk].astype(int), sc[mk][None])[0])
                else:
                    row["spearman"], row["p_perm"] = perm_spearman(sc[mk], x[mk], RV["n_perm"], rng)
                clin.append(row)
        print(name, flush=True)
    A, C = pd.DataFrame(aucs), pd.DataFrame(clin)
    A.to_csv(O / "auroc.tsv", sep="\t", index=False)
    C.to_csv(O / "clinical.tsv", sep="\t", index=False)
    write_provenance("21_robustness/A3_sensitivity/indep", [ROOT / "results/19_shared_program/kpmp/core_procurement.tsv",
                                                        ROOT / "results/19_shared_program/core/core_genes.tsv"],
                     [O / "auroc.tsv", O / "clinical.tsv"], CFG["seed"], {"variants": RV["gene_variants"]})
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(A.pivot_table(index="cohort", columns="variant", values="auroc").round(3).to_string())
        print(C.pivot_table(index=["cohort", "test"], columns="variant", values=["spearman", "auroc"]).round(3).to_string())


if __name__ == "__main__":
    main()
