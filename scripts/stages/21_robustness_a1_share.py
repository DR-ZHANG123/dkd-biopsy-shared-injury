"""Stage 21 A1（+A3 bulk 部分、A5 余弦零分布）：疾病 vs 对照 逐基因效应中由外部 injury–repair 方向解释的份额。

每个评估单元 u（config injury.compartments.*.eval）与其中每个诊断 k（n ≥ robustness21.min_group，同 stage 11）：
  a      = stage 11 的外部方向（results/11_injury/axis_<comp>_<u>.tsv；只用其他来源单元的 非 DKD 病人 vs 对照）
  shift  = 秩均值差（与 stage 11 cosine 同口径）；g = Hedges g（秩尺度）
  cosine(shift, a) 与 cos²；R²_raw = Pearson(g, a)²；
  R²_disatt = R²_raw / (rel_g · rel_a)：rel_g = 1 − mean(v_g)/var(g)（v_g = Hedges g 抽样方差），
              rel_a = 1 − mean(v_a)/var(a)（v_a = 来源单元抽样方差之和 / k²）；即「可复现（非抽样噪声）部分」中被 a 解释的比例。
  零分布：单元内 k 与对照的样本标签置换；两组线性模型的随机旋转（保留基因间相关）。报告观测、零分布中位数、95% 分位与经验 P。
  DE：单元内 k vs 对照 Welch t（秩）BH-FDR < de_fdr 的基因中，属于 stage 19 反应基因（该单元为来源时用去掉本单元后重估的
      同大小集合）且方向一致的比例；随机期望 = 反应基因占基因宇宙比例 × 1/2 的上限参考；另给显著基因与 a 同号比例。
A3：gene_variants（full / no_proc / no_ieg / no_proc_ieg）逐一剔除基因后重算上述量，以及
    stage 11 分数（剔除后在 a 上重新取 top_k）与 stage 19 反应基因分数的 DKD vs 对照 AUROC。
另：跨队列可复现份额（replicable_share）——同一诊断在两个不共享标本的单元中，外部方向（或另一单元的 非 DKD 效应）
    解释的「跨队列共享信号」比例（见 replicable_share 文档）。零分布说明：ctrlsplit（只在对照内拆组，主）、residrot（组内残差旋转）、
    labelperm（疾病 / 对照标签置换；存在真实效应时偏保守）。
产出 results/21_robustness/A1_share/{share.tsv, replicable_share.tsv, dkd_auroc.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.injury import NON_DISEASE, effect_sizes, full_ranks  # noqa: E402
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.robustness21 import (CFG, OUT, RV, cosine, core_exclude, ctrl_split_null, exclude, null_summary,  # noqa: E402
                          perm_null, r2, resid_rotation_null, two_group, units, welch_bh)
from lib.scp import signed_score  # noqa: E402
from lib.scp_core import loo_sets  # noqa: E402
from lib.stats import auc_rows  # noqa: E402

O = OUT / "A1_share"
INJ = CFG["injury"]


def axis_noise(comp: str, u: str, s, U, genes: pd.Index, a: pd.Series) -> np.ndarray:
    """外部方向每个基因的抽样方差（来源单元 非 DKD vs 对照 的 d 方差之和 / k²）。"""
    vs = []
    for src in INJ["compartments"][comp]["source"]:
        if src == u:
            continue
        d = s.loc[U[src], "diagnosis"]
        n1, n2 = int((~d.isin(NON_DISEASE)).sum()), int((d == "CONTROL").sum())
        if n2 < INJ["min_controls"] or n1 < INJ["min_disease"]:
            continue
        R = full_ranks(U[src], s).dropna(axis=1)
        ok = pd.Series(1.0, index=R.columns).reindex(genes)
        v = (n1 + n2) / (n1 * n2) + a.to_numpy() ** 2 / (2 * (n1 + n2))
        vs.append(np.where(ok.notna().to_numpy(), v, np.nan))
    V = np.vstack(vs)
    k = np.isfinite(V).sum(0)
    return np.nansum(V, 0) / np.maximum(k, 1) ** 2


def contrast_rows(comp, u, s, U, rng) -> tuple[list[dict], dict]:
    a_all = pd.read_csv(ROOT / f"results/11_injury/axis_{comp}_{u}.tsv", sep="\t", index_col=0).injury_effect
    R = full_ranks(U[u], s).dropna(axis=1)
    genes = R.columns.intersection(a_all.index)
    R, a_all = R[genes], a_all[genes]
    va_all = axis_noise(comp, u, s, U, genes, a_all)
    diag = s.loc[R.index, "diagnosis"].to_numpy()
    ctrl = diag == "CONTROL"
    up, dn = loo_sets(comp, u, s, U)
    resp = pd.Series(0, index=genes)
    resp[resp.index.isin(up)], resp[resp.index.isin(dn)] = 1, -1
    rows, gvec = [], {}
    for k in sorted(set(diag) - (NON_DISEASE - {"DKD"})):
        m = diag == k
        if m.sum() < RV["min_group"] or ctrl.sum() < RV["min_group"]:
            continue
        Y_all = np.vstack([R.to_numpy()[m], R.to_numpy()[ctrl]])
        gvec[(u, k)] = pd.Series(two_group(Y_all[:m.sum()], Y_all[m.sum():])[0], index=genes)
        for var in RV["gene_variants"]:
            keep = ~genes.isin(exclude(var, a_all))
            Y, a, va = Y_all[:, keep], a_all.to_numpy()[keep], va_all[keep]
            n1 = int(m.sum())
            g, v, sh = two_group(Y[:n1], Y[n1:])
            rel_g = 1 - v.mean() / g.var()
            rel_a = 1 - va.mean() / a.var()
            row = {"compartment": comp, "unit": u, "diagnosis": k, "variant": var, "n": n1, "n_ctrl": int(ctrl.sum()),
                   "n_genes": int(keep.sum()), "rel_g": rel_g, "rel_a": rel_a}
            cos_o, r2_o = cosine(sh, a), r2(g, a)
            row["r2_disatt"] = r2_o / (rel_g * rel_a) if rel_g > 0 and rel_a > 0 else np.nan
            nulls = (("ctrlsplit", ctrl_split_null(Y[n1:], n1, RV["n_perm"], rng)),
                     ("residrot", resid_rotation_null(Y, n1, RV["n_rot"], rng)),
                     ("labelperm", perm_null(Y, n1, RV["n_perm"], rng)))
            for tag, gen in nulls:
                cs, rs = [], []
                for G, SH in gen:
                    an = a / np.linalg.norm(a)
                    cs.append(np.abs(SH @ an) / np.linalg.norm(SH, axis=1))
                    Gc = G - G.mean(1, keepdims=True)
                    ac = a - a.mean()
                    rs.append(((Gc @ ac) / (np.linalg.norm(Gc, axis=1) * np.linalg.norm(ac))) ** 2)
                cs, rs = np.concatenate(cs), np.concatenate(rs)
                row |= null_summary(f"cosine_{tag}", cos_o, cs) | null_summary(f"r2_{tag}", r2_o, rs)
                row[f"r2_disatt_{tag}_null_q95"] = float(np.quantile(rs, 0.95) / (rel_g * rel_a)) if rel_g > 0 else np.nan
            row["cos2"] = cos_o ** 2
            _, q = welch_bh(Y[:n1], Y[n1:])
            sig = q < RV["de_fdr"]
            rr = resp.to_numpy()[keep]
            sgn = np.sign(g)
            conc = sig & (rr != 0) & (np.sign(rr) == sgn)
            row |= {"n_de": int(sig.sum()), "n_de_response_concordant": int(conc.sum()),
                    "frac_de_response_concordant": float(conc.sum() / sig.sum()) if sig.sum() else np.nan,
                    "frac_de_response_discordant": float((sig & (rr != 0) & (np.sign(rr) != sgn)).sum() / sig.sum())
                    if sig.sum() else np.nan,
                    "frac_response_in_universe": float((rr != 0).mean()),
                    "frac_de_sign_agree_axis": float((np.sign(a[sig]) == sgn[sig]).mean()) if sig.sum() else np.nan,
                    "frac_response_that_are_de": float(conc.sum() / max((rr != 0).sum(), 1))}
            rows.append(row)
    return rows, gvec


def dkd_auroc(comp, u, s, U) -> list[dict]:
    a_all = pd.read_csv(ROOT / f"results/11_injury/axis_{comp}_{u}.tsv", sep="\t", index_col=0).injury_effect
    R = full_ranks(U[u], s).dropna(axis=1)
    d = s.loc[R.index, "diagnosis"].to_numpy()
    m = np.isin(d, ["DKD", "CONTROL"])
    y = (d[m] == "DKD").astype(int)
    if y.sum() < 3 or (1 - y).sum() < 3:
        return []
    up, dn = loo_sets(comp, u, s, U)
    out = []
    for var in RV["gene_variants"]:
        ax = a_all.reindex(R.columns).dropna()
        ax = ax[~ax.index.isin(exclude(var, ax))]
        k = INJ["top_k"]
        sc11 = (R[ax.nlargest(k).index].mean(1) - R[ax.nsmallest(k).index].mean(1)).to_numpy()
        ex = core_exclude(var, comp)
        u2, d2 = [g for g in up if g not in ex], [g for g in dn if g not in ex]
        sc19 = signed_score(R, u2, d2).to_numpy()
        out.append({"compartment": comp, "unit": u, "variant": var, "n_dkd": int(y.sum()), "n_ctrl": int((1 - y).sum()),
                    "auroc_stage11_score": float(auc_rows(y, sc11[m][None])[0]),
                    "auroc_response_gene_score": float(auc_rows(y, sc19[m][None])[0]),
                    "n_up_response": len(u2), "n_down_response": len(d2)})
    return out


def replicable_share(G: dict, O_: dict, A: dict, relA: dict, s, U, rng, comp: str) -> list[dict]:
    """同一诊断在两个不共享标本的评估单元 u、v 中：DKD（或该诊断）效应 g_u = s + c_u + e_u。
    cov(g_u, g_v) 估计跨队列共享信号方差 var(s)；外部方向 a_u（不含 u）解释的共享信号比例
      share_a = cov(g_u, a_u)² / (var(a_u) · rel_a · cov(g_u, g_v))；
    同理以单元 v 的 非 DKD 病人 vs 对照 效应 o_v 代替 a_u：share_o = cov(g_u, o_v)² / (var(o_v) · rel_o · cov(g_u, g_v))。
    bootstrap：u、v 内按组重抽样病人（a_u 固定）。"""
    rows = []
    keys = sorted(G)
    for (u, k) in keys:
        for (v, k2) in keys:
            if k2 != k or v == u:
                continue
            has_o = v in O_
            gi = G[(u, k)]["g"].index.intersection(G[(v, k)]["g"].index).intersection(A[u].index)
            if has_o:
                gi = gi.intersection(O_[v]["g"].index)

            def est(gu, gv, ov):
                a = A[u][gi].to_numpy()
                c_uv = np.cov(gu, gv)[0, 1]
                sa = np.cov(gu, a)[0, 1] ** 2 / (a.var(ddof=1) * relA[u] * c_uv)
                so = np.cov(gu, ov)[0, 1] ** 2 / (ov.var(ddof=1) * O_[v]["rel"] * c_uv) if ov is not None else np.nan
                return c_uv, sa, so
            obs = est(G[(u, k)]["g"][gi].to_numpy(), G[(v, k)]["g"][gi].to_numpy(),
                      O_[v]["g"][gi].to_numpy() if has_o else None)
            bs = []
            for _ in range(RV["n_boot"] // 5):
                gu = boot_g(G[(u, k)], gi, rng)
                gv = boot_g(G[(v, k)], gi, rng)
                bs.append(est(gu, gv, boot_g(O_[v], gi, rng) if has_o else None))
            bs = np.array(bs, float)
            rows.append({"compartment": comp, "unit_u": u, "unit_v": v, "diagnosis": k, "n_genes": len(gi),
                         "cov_uv_shared_signal": obs[0], "r_uv": float(np.corrcoef(G[(u, k)]["g"][gi], G[(v, k)]["g"][gi])[0, 1]),
                         "share_by_external_axis": obs[1], "share_by_external_axis_lo": np.percentile(bs[:, 1], 2.5),
                         "share_by_external_axis_hi": np.percentile(bs[:, 1], 97.5),
                         "share_by_nonDKD_v": obs[2], "share_by_nonDKD_v_lo": np.nanpercentile(bs[:, 2], 2.5) if has_o else np.nan,
                         "share_by_nonDKD_v_hi": np.nanpercentile(bs[:, 2], 97.5) if has_o else np.nan})
    return rows


def boot_g(d: dict, gi: pd.Index, rng) -> np.ndarray:
    A, B = d["A"], d["B"]
    g, _, _ = two_group(A[rng.choice(len(A), len(A))], B[rng.choice(len(B), len(B))])
    return pd.Series(g, index=d["genes"])[gi].to_numpy()


def unit_groups(comp, u, s, U):
    """单元 u 的 R、各诊断 vs 对照 的 g 与原始矩阵（bootstrap 用），以及 非 DKD 病人 vs 对照 的 g 与 rel。"""
    R = full_ranks(U[u], s).dropna(axis=1)
    d = s.loc[R.index, "diagnosis"].to_numpy()
    X, ctrl = R.to_numpy(), d == "CONTROL"
    out = {}
    for k in sorted(set(d) - (NON_DISEASE - {"DKD"})):
        m = d == k
        if m.sum() >= RV["min_group"] and ctrl.sum() >= RV["min_group"]:
            g, v, _ = two_group(X[m], X[ctrl])
            out[k] = {"g": pd.Series(g, index=R.columns), "A": X[m], "B": X[ctrl], "genes": R.columns,
                      "rel": 1 - v.mean() / g.var()}
    m = ~np.isin(d, list(NON_DISEASE))
    g, v, _ = two_group(X[m], X[ctrl]) if m.sum() >= 3 else (None, None, None)
    oth = None if g is None else {"g": pd.Series(g, index=R.columns), "A": X[m], "B": X[ctrl], "genes": R.columns,
                                  "rel": 1 - v.mean() / g.var()}
    return out, oth


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, U = units()
    rows, reps, aucs = [], [], []
    for comp, spec in INJ["compartments"].items():
        G, O_, A, relA = {}, {}, {}, {}
        for u in spec["eval"]:
            r, _ = contrast_rows(comp, u, s, U, rng)
            rows += r
            aucs += dkd_auroc(comp, u, s, U)
            gd, oth = unit_groups(comp, u, s, U)
            G |= {(u, k): v for k, v in gd.items()}
            if oth is not None:
                O_[u] = oth
            A[u] = pd.read_csv(ROOT / f"results/11_injury/axis_{comp}_{u}.tsv", sep="\t", index_col=0).injury_effect
            relA[u] = float(pd.DataFrame(r).query("variant == 'full'").rel_a.iloc[0])
            print(comp, u, "done", flush=True)
        reps += replicable_share(G, O_, A, relA, s, U, rng, comp)
    T, P, A = pd.DataFrame(rows), pd.DataFrame(reps), pd.DataFrame(aucs)
    ref = pd.read_csv(ROOT / "results/11_injury/shift_decomposition.tsv", sep="\t")
    chk = T[T.variant == "full"].merge(ref, left_on=["unit", "diagnosis"], right_on=["eval_unit", "diagnosis"])
    assert np.allclose(chk.cosine_ctrlsplit, chk.cosine, atol=1e-4), "与 stage 11 cosine 不一致"
    outs = {"share.tsv": T, "replicable_share.tsv": P, "dkd_auroc.tsv": A}
    for f, t in outs.items():
        t.to_csv(O / f, sep="\t", index=False)
    write_provenance("21_robustness/A1_share", [ROOT / "data/processed/samples.tsv", ROOT / "results/11_injury/shift_decomposition.tsv",
                                            ROOT / "results/19_shared_program/core/core_genes.tsv",
                                            ROOT / "results/19_shared_program/kpmp/procurement.tsv.gz"],
                     [O / f for f in outs], CFG["seed"], {"n_perm": RV["n_perm"], "n_rot": RV["n_rot"], "de_fdr": RV["de_fdr"]})
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        cols = ["unit", "diagnosis", "variant", "n", "cosine_ctrlsplit", "cosine_ctrlsplit_null_q95", "cosine_residrot_null_q95",
                "cosine_labelperm_null_q95", "r2_ctrlsplit", "r2_ctrlsplit_null_q95", "r2_residrot_null_q95", "rel_g", "rel_a", "r2_disatt", "n_de", "frac_de_response_concordant"]
        print(T[T.variant == "full"][cols].round(3).to_string())
        print(P.round(3).to_string())
        print(A.round(3).to_string())


if __name__ == "__main__":
    main()
