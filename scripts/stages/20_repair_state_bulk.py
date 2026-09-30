"""Stage 20e：bulk 肾小管单元中的修复失败状态 —— BayesPrism 状态层份额与状态程序分数：各病种 vs 对照（跨队列一致性）与 SCP 的关系。

度量（每样本）：
  share_<谱系>:<状态>   BayesPrism θ_state（主参照 scRNA，stage 16 run/all）中该状态占 PT（或 TAL）类型的份额（logit 后检验）
  prog_<变体>_<程序>    状态程序（stage 20a，snRNA 定义）上调 − 下调 的样本内秩分数；noSCP = 剔除全部 SCP 核心基因
  progup_<变体>_<程序>  只用上调基因
单元 = repair_state.bulk.units（来源单元 + 其他肾小管 / 全肾单元）；每病种 vs 对照（组 ≥ min_group）：Hedges g、AUROC、置换 P；
跨单元 DerSimonian–Laird 合并与同号比例。与 SCP：单元内 Spearman（全体、病人内）；SCP 分数在来源单元用去掉本单元后重估的核心集。
可分辨性：θ 份额与程序分数的单元内 Spearman（两种独立的 bulk 读数是否一致）、份额中位数与近零比例。
产出 results/20_repair_state/bulk/
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
from lib.repair_bulk import hedges_var, program_scores, theta_shares  # noqa: E402
from lib.repair_state import CFG, OUT, RS, logit, perm_spearman  # noqa: E402
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.scp import signed_score  # noqa: E402
from lib.scp_core import loo_sets  # noqa: E402
from lib.stats import dersimonian_laird  # noqa: E402

O = OUT / "bulk"
B = RS["bulk"]


def unit_measures(u: str, s, units) -> pd.DataFrame:
    idx = units[u]
    R = full_ranks(idx, s).dropna(axis=1)
    up, dn = loo_sets(RS["compartment"], u, s, units)
    M = pd.DataFrame({"scp": signed_score(R, up, dn)})
    M = M.join(program_scores(R)).join(theta_shares(R.index))
    for c in [c for c in M if c.startswith("share_")]:
        M[c] = logit(M[c].to_numpy(float), B["eps"])
    M["diagnosis"] = s.loc[M.index, "diagnosis"].to_numpy()
    M["unit"] = u
    return M


def contrast_rows(M: pd.DataFrame, rng) -> list[dict]:
    rows = []
    meas = [c for c in M if c.startswith(("share_", "prog", "scp", "theta_"))]
    d = M.diagnosis.to_numpy()
    ctrl = d == "CONTROL"
    tests = {k: d == k for k in B["diseases"]}
    tests["nonDKD_patients"] = ~np.isin(d, ["CONTROL", "DKD", "OTHER", "UNKNOWN", "TMD"])
    for k, pos in tests.items():
        if pos.sum() < B["min_group"] or ctrl.sum() < B["min_group"]:
            continue
        for c in meas:
            x = M[c].to_numpy(float)
            a, b = x[pos & np.isfinite(x)], x[ctrl & np.isfinite(x)]
            if len(a) < B["min_group"] or len(b) < B["min_group"]:
                continue
            sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            g = (1 - 3 / (4 * (len(a) + len(b)) - 9)) * (a.mean() - b.mean()) / (sd + 1e-12)
            xx = np.r_[a, b]
            null = np.array([(lambda p: p[:len(a)].mean() - p[len(a):].mean())(rng.permutation(xx))
                             for _ in range(RS["n_perm"])])
            obs = a.mean() - b.mean()
            rows.append({"unit": M.unit.iloc[0], "disease": k, "measure": c, "n_pos": len(a), "n_ctrl": len(b),
                         "hedges_g": g, "var_g": hedges_var(g, len(a), len(b)),
                         "auroc": stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)),
                         "p_perm": (1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (1 + len(null))})
    return rows


def meta_rows(C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (k, c), g in C.groupby(["disease", "measure"]):
        r = dersimonian_laird(g.hedges_g.to_numpy(float), g.var_g.to_numpy(float))
        rows.append({"disease": k, "measure": c, "n_units": len(g), "frac_units_positive": (g.hedges_g > 0).mean(),
                     "median_auroc": g.auroc.median(), **{f"re_{x}": v for x, v in r.items()}})
    return pd.DataFrame(rows)


def scp_relation(M: pd.DataFrame, rng) -> list[dict]:
    rows = []
    d = M.diagnosis.to_numpy()
    pat = d != "CONTROL"
    for c in [c for c in M if c.startswith(("share_", "prog", "theta_"))]:
        for sub, m in (("all", np.ones(len(M), bool)), ("patients", pat)):
            x, y = M[c].to_numpy(float)[m], M.scp.to_numpy(float)[m]
            if np.isfinite(x).sum() < 10:
                continue
            r, p = perm_spearman(x, y, RS["n_perm"], rng)
            rows.append({"unit": M.unit.iloc[0], "measure": c, "samples": sub, "n": int(np.isfinite(x).sum()),
                         "spearman_vs_scp": r, "p_perm": p})
    return rows


def resolvability(M: pd.DataFrame) -> list[dict]:
    rows = []
    for c in [c for c in M if c.startswith("share_")]:
        lin_st = c.split("_", 1)[1]
        raw = 1 / (1 + np.exp(-M[c].to_numpy(float)))
        row = {"unit": M.unit.iloc[0], "state": lin_st, "n": int(np.isfinite(raw).sum()),
               "median_share": float(np.nanmedian(raw)), "frac_below_1pct": float(np.nanmean(raw < 0.01))}
        for v in ("full", "noSCP"):
            pc = f"prog_{v}_{lin_st}"
            if pc in M:
                ok = np.isfinite(M[c]) & np.isfinite(M[pc])
                row[f"spearman_share_vs_prog_{v}"] = stats.spearmanr(M[c][ok], M[pc][ok])[0] if ok.sum() > 10 else np.nan
        rows.append(row)
    return rows


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, units = import_module("11_injury_axis").load_samples()
    allM, con, rel, res = [], [], [], []
    for u in B["units"]:
        M = unit_measures(u, s, units)
        allM.append(M.rename_axis("sample_uid").reset_index())
        con += contrast_rows(M, rng)
        rel += scp_relation(M, rng)
        res += resolvability(M)
        print(u, len(M), flush=True)
    C = pd.DataFrame(con)
    outs = {"sample_measures.tsv.gz": pd.concat(allM, ignore_index=True), "unit_contrasts.tsv": C,
            "meta_contrasts.tsv": meta_rows(C), "scp_relation.tsv": pd.DataFrame(rel),
            "resolvability.tsv": pd.DataFrame(res)}
    for f, t in outs.items():
        t.to_csv(O / f, sep="\t", index=False)
    write_provenance("20_repair_state/bulk", [ROOT / "results/16_deconv/run" / B["theta_run"] / "theta_state.tsv",
                                              OUT / "programs/program_sets.tsv"],
                     [O / f for f in outs], CFG["seed"], {"bulk": B})
    key = ["share_PT:rfPT", "share_TAL:rfTAL", "prog_full_PT:rfPT", "prog_noSCP_PT:rfPT", "prog_full_TAL:rfTAL",
           "prog_noSCP_TAL:rfTAL", "progup_noSCP_PT:rfPT", "scp"]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        m = outs["meta_contrasts.tsv"]
        print(m[m.measure.isin(key)][["disease", "measure", "n_units", "frac_units_positive", "median_auroc", "re_mu",
                                      "re_lo", "re_hi", "re_I2", "re_p"]].round(3).to_string())
        r = outs["scp_relation.tsv"]
        print(r[r.measure.isin(key)].round(3).to_string())
        print(outs["resolvability.tsv"].round(3).to_string())


if __name__ == "__main__":
    main()
