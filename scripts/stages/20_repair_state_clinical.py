"""Stage 20f：修复失败状态读数在独立队列中的复现与临床关联（只作检验，不参与任何定义；每个检验只运行一次）。

队列 = repair_state.bulk.clinical_cohorts；样本、临床字段与 stage 19d 同口径（复用其 load_cohort / clinical）。
度量：状态程序分数（full / noSCP / 只上调）；有 BayesPrism θ 的队列（GSE175759、GSE142025、GSE115857）另报 θ 份额。
检验：疾病 vs 对照 AUROC（该队列 stage 19d 的阳性病种）；临床变量 Spearman（置换 P）；偏 Spearman（控制 SCP 分数，秩残差）
      —— 回答「修复失败读数是否携带 SCP 之外的临床信息」。GSE142025 晚期 vs 早期为 AUROC。
KPMP 切片 bulk 与 KPMP 单细胞图谱有部分参与者重叠：另报去掉重叠参与者后的敏感性结果。
产出 results/20_repair_state/clinical/
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
from lib.repair_bulk import program_scores, theta_shares  # noqa: E402
from lib.repair_state import CFG, OUT, RS, logit, perm_spearman  # noqa: E402
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.scp import signed_score  # noqa: E402
from lib.scp_core import load_core  # noqa: E402

O = OUT / "clinical"
KEY = ["prog_full_PT:rfPT", "prog_noSCP_PT:rfPT", "progup_noSCP_PT:rfPT", "prog_full_TAL:rfTAL", "prog_noSCP_TAL:rfTAL",
       "prog_noSCP_PT:aPT", "prog_noSCP_PT:frPT", "prog_noSCP_TAL:aTAL", "prog_noSCP_TAL:frTAL",
       "share_PT:rfPT", "share_PT:aPT", "share_PT:frPT", "share_TAL:rfTAL", "scp"]


def rank_resid(v, c):
    rv, rc = stats.rankdata(v), stats.rankdata(c)
    X = np.c_[np.ones(len(rc)), rc]
    return rv - X @ np.linalg.lstsq(X, rv, rcond=None)[0]


def atlas_donors() -> set[str]:
    out = set()
    for n in RS["datasets"]:
        d = pd.read_csv(ROOT / CFG["kpmp"]["pseudobulk_cache"] / n / "donors.tsv", sep="\t", dtype={"donor": str})
        out |= set(d.donor)
    return out


def measures(name, R, meta) -> pd.DataFrame:
    up, dn = load_core(RS["compartment"])
    M = pd.DataFrame({"scp": signed_score(R, [g for g in up if g in R], [g for g in dn if g in R])})
    M = M.join(program_scores(R)).join(theta_shares(R.index))
    for c in [c for c in M if c.startswith("share_")]:
        M[c] = logit(M[c].to_numpy(float), RS["bulk"]["eps"])
    return M


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    S19 = import_module("19_shared_program_replicate")
    s, units = import_module("11_injury_axis").load_samples()
    spec_all = CFG["shared_program"]["replicate"]["cohorts"]
    overlap = atlas_donors()
    rows, scores = [], []
    for name in RS["bulk"]["clinical_cohorts"]:
        spec = spec_all[name]
        R, meta = S19.load_cohort(name, spec, s, units)
        M = measures(name, R, meta)
        d = meta.diagnosis.to_numpy()
        variants = [("all", np.ones(len(M), bool))]
        if name == "KPMP":
            ov = meta.participant.astype(str).isin(overlap).to_numpy()
            variants.append(("no_atlas_overlap", ~ov))
            M["in_kpmp_atlas"] = ov
        scores.append(M.assign(cohort=name, diagnosis=d).rename_axis("sample_uid").reset_index())
        cols = [c for c in KEY if c in M and M[c].notna().sum() >= 5]
        for vname, vm in variants:
            m = np.isin(d, spec["positive"] + ["CONTROL"]) & vm
            y = np.isin(d[m], spec["positive"]).astype(int)
            for c in cols:
                x = M[c].to_numpy(float)
                if y.sum() >= 3 and (1 - y).sum() >= 3:
                    ok = np.isfinite(x[m])
                    a = float(stats.mannwhitneyu(x[m][ok][y[ok] == 1], x[m][ok][y[ok] == 0]).statistic
                              / ((y[ok] == 1).sum() * (y[ok] == 0).sum()))
                    rows.append({"cohort": name, "samples": vname, "test": "disease_vs_control", "measure": c,
                                 "n": int(ok.sum()), "n_pos": int(y[ok].sum()), "auroc": a})
                for test, mk, var in S19.clinical(name, meta):
                    mk = mk & vm & np.isfinite(x)
                    if mk.sum() < 8:
                        continue
                    row = {"cohort": name, "samples": vname, "test": test, "measure": c, "n": int(mk.sum())}
                    if test == "advanced_vs_early_DN":
                        yy = var[mk].astype(int)
                        row["auroc"] = float(stats.mannwhitneyu(x[mk][yy == 1], x[mk][yy == 0]).statistic
                                             / ((yy == 1).sum() * (yy == 0).sum()))
                        row["n_pos"] = int(yy.sum())
                        if c != "scp":
                            rx = rank_resid(x[mk], M.scp.to_numpy(float)[mk])
                            row["auroc_given_scp_resid"] = float(stats.mannwhitneyu(rx[yy == 1], rx[yy == 0]).statistic
                                                                 / ((yy == 1).sum() * (yy == 0).sum()))
                    else:
                        r, p = perm_spearman(x[mk], var[mk], RS["n_perm"], rng)
                        row |= {"spearman": r, "p_perm": p}
                        if c != "scp":
                            sc = M.scp.to_numpy(float)[mk]
                            pr, pp = perm_spearman(rank_resid(x[mk], sc), rank_resid(var[mk], sc), RS["n_perm"], rng)
                            row |= {"partial_spearman_given_scp": pr, "p_perm_partial": pp}
                    rows.append(row)
        print(name, len(M), flush=True)
    T = pd.DataFrame(rows)
    outs = {"tests.tsv": T, "scores.tsv.gz": pd.concat(scores, ignore_index=True)}
    for f, t in outs.items():
        t.to_csv(O / f, sep="\t", index=False)
    write_provenance("20_repair_state/clinical", [OUT / "programs/program_sets.tsv",
                                                  ROOT / "results/17_independent/test_samples.tsv"],
                     [O / f for f in outs], CFG["seed"], {"cohorts": RS["bulk"]["clinical_cohorts"], "n_perm": RS["n_perm"]})
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(T.round(3).to_string())


if __name__ == "__main__":
    main()
