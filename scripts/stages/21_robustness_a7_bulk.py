"""Stage 21 A7（bulk / 独立队列部分）：修复失败程序与 injury–repair 分数的循环性。

分数变体：scp_full（stage 19 TUB 反应基因；来源单元用去掉本单元后重估的集合，同 stage 20e）、
          scp_no_driver（剔除 stage 20g 任一状态的修复失败驱动基因）。
程序：prog_noSCP_<状态>（stage 20a 打分程序剔除全部反应基因，同 stage 20e/f）与 prog_REF_noSCP_<状态>（只用 REF 供体推导的程序，
      剔除全部反应基因；CKD 供体不参与程序定义）。两类程序与两种分数在基因上完全不相交。
(1) bulk 单元（repair_state.bulk.units）内：程序 vs 分数的 Spearman（全体 / 病人内）。
(2) 独立临床队列（repair_state.bulk.clinical_cohorts，与 stage 20f 同样本与临床字段）：分数与程序各自的临床 Spearman；
    程序对临床变量的偏 Spearman（控制 scp_full 或 scp_no_driver；秩残差）；GSE142025 晚期 vs 早期为 AUROC 及残差 AUROC。
产出 results/21_robustness/A7_circularity/bulk/{unit_relation.tsv, clinical.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.injury import full_ranks  # noqa: E402
from lib.repair_bulk import program_sets  # noqa: E402
from lib.repair_state import OUT as RS_OUT, RS, perm_spearman  # noqa: E402
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.robustness21 import CFG, OUT, RV, units  # noqa: E402
from lib.scp import signed_score  # noqa: E402
from lib.scp_core import load_core, load_core_table, loo_sets  # noqa: E402

O = OUT / "A7_circularity" / "bulk"
KPMP21 = import_module("21_robustness_kpmp")


def rank_resid(v, c):
    rv, rc = stats.rankdata(v), stats.rankdata(c)
    X = np.c_[np.ones(len(rc)), rc]
    return rv - X @ np.linalg.lstsq(X, rv, rcond=None)[0]


def measures(R: pd.DataFrame, up, dn, drv, progs) -> pd.DataFrame:
    keep = lambda g: [x for x in g if x in R.columns]  # noqa: E731
    M = pd.DataFrame({"scp_full": signed_score(R, keep(up), keep(dn)),
                      "scp_no_driver": signed_score(R, keep([g for g in up if g not in drv]), keep([g for g in dn if g not in drv]))})
    for k, (u, d) in progs.items():
        u, d = keep(u), keep(d)
        if len(u) >= 10 and len(d) >= 10:
            M[k] = R[u].mean(1) - R[d].mean(1)
    return M


def auc(x, y):
    return float(stats.mannwhitneyu(x[y == 1], x[y == 0]).statistic / ((y == 1).sum() * (y == 0).sum()))


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, U = units()
    S19 = import_module("19_shared_program_replicate")
    drv = KPMP21.drivers()
    all_core = set(load_core_table().gene)
    progs = {f"prog_noSCP_{k}": v for k, v in program_sets("noSCP").items() if k in RV["a7_programs"]}
    progs |= {f"prog_REF_noSCP_{k}": v for k, v in KPMP21.ref_programs(all_core).items()}
    comp = RS["compartment"]
    rel = []
    for u in RS["bulk"]["units"]:
        R = full_ranks(U[u], s).dropna(axis=1)
        up, dn = loo_sets(comp, u, s, U)
        M = measures(R, up, dn, drv, progs)
        pat = (s.loc[R.index, "diagnosis"] != "CONTROL").to_numpy()
        for sub, m in (("all", np.ones(len(M), bool)), ("patients", pat)):
            for p in [c for c in M if c.startswith("prog")]:
                for sc in ("scp_full", "scp_no_driver"):
                    r, pp = perm_spearman(M[p].to_numpy()[m], M[sc].to_numpy()[m], RS["n_perm"], rng)
                    rel.append({"unit": u, "samples": sub, "program": p, "score": sc, "n": int(m.sum()), "spearman": r, "p_perm": pp})
        print(u, flush=True)
    up, dn = load_core(comp)
    clin = []
    for name in RS["bulk"]["clinical_cohorts"]:
        spec = CFG["shared_program"]["replicate"]["cohorts"][name]
        R, meta = S19.load_cohort(name, spec, s, U)
        M = measures(R, up, dn, drv, progs)
        d = meta.diagnosis.to_numpy()
        m = np.isin(d, spec["positive"] + ["CONTROL"])
        y = np.isin(d[m], spec["positive"]).astype(int)
        for c in M:
            x = M[c].to_numpy(float)
            row = {"cohort": name, "test": "disease_vs_control", "measure": c, "n": int(m.sum()), "auroc": auc(x[m], y)}
            clin.append(row)
            for test, mk, var in S19.clinical(name, meta):
                mk = mk & np.isfinite(x)
                if mk.sum() < 8:
                    continue
                row = {"cohort": name, "test": test, "measure": c, "n": int(mk.sum())}
                if test == "advanced_vs_early_DN":
                    yy = var[mk].astype(int)
                    row["auroc"] = auc(x[mk], yy)
                    if c.startswith("prog"):
                        for sc in ("scp_full", "scp_no_driver"):
                            row[f"auroc_resid_given_{sc}"] = auc(rank_resid(x[mk], M[sc].to_numpy()[mk]), yy)
                else:
                    row["spearman"], row["p_perm"] = perm_spearman(x[mk], var[mk], RS["n_perm"], rng)
                    if c.startswith("prog"):
                        for sc in ("scp_full", "scp_no_driver"):
                            z = M[sc].to_numpy()[mk]
                            row[f"partial_given_{sc}"], row[f"p_perm_partial_{sc}"] = perm_spearman(
                                rank_resid(x[mk], z), rank_resid(var[mk], z), RS["n_perm"], rng)
                clin.append(row)
        print(name, flush=True)
    A, C = pd.DataFrame(rel), pd.DataFrame(clin)
    A.to_csv(O / "unit_relation.tsv", sep="\t", index=False)
    C.to_csv(O / "clinical.tsv", sep="\t", index=False)
    write_provenance("21_robustness/A7_circularity/bulk", [RS_OUT / "genes/driver_calls.tsv", RS_OUT / "programs/program_sets.tsv",
                                                       RS_OUT / "programs/programs.tsv.gz"],
                     [O / "unit_relation.tsv", O / "clinical.tsv"], CFG["seed"], {"programs": list(progs)})
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(A[A.samples == "patients"].pivot_table(index=["unit"], columns=["program", "score"], values="spearman").round(2).T.to_string())
        print(C.round(3).to_string())


if __name__ == "__main__":
    main()
