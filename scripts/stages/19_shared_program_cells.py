"""Stage 19b：SCP 的关键细胞 —— (a) KPMP 健康参考中的细胞类型定位；(b) BayesPrism 组成对 SCP 分数的解释。

(a) snRNA / scRNA 健康参考供体（REF）× 细胞类型（kpmp.map_to_stage13）pseudobulk 的均值 log2CPM：
    每基因最高表达类型、tau、最高类型份额（复用 stage 18d localisation）；核心上 / 下调基因按最高表达类型的
    富集（Fisher，背景 = 同区室 meta 全部表达基因）；KPMP 标志基因集上 SCP g_re 的均值 z（基因置换）。
(b) 每个单元：SCP 分数（来源单元用去掉本单元后重估的核心集）~ 组成（CLR θ，stage 16 run/all）的 R² 与
    5 折 CV R²（同大小随机带符号基因集为对照）；各类型比例与 SCP 的 Spearman（全体 / 病人内）；
    疾病 vs 对照 AUROC：原始分数 vs 组成调整后（CV 折外残差）。
产出 results/19_shared_program/cells/
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.injury import full_ranks  # noqa: E402
from lib.kpmp import Pseudobulk  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from lib.scp import bh, clr, patient_mask, signed_score  # noqa: E402
from lib.scp_core import OUT, SP, load_core_table, loo_sets  # noqa: E402
from lib.stats import auc_rows  # noqa: E402

CFG = load_config()
K = CFG["kpmp"]
CO = SP["composition"]
O = OUT / "cells"


def localise(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    S18 = import_module("18_interpret_kpmp")
    metas = {c: pd.read_csv(OUT / "core" / f"meta_{c}.tsv.gz", sep="\t") for c in SP["sources"]}
    genes = sorted(set().union(*[set(m.gene) for m in metas.values()]))
    locs, enr = [], []
    for name in SP["kpmp"]["datasets"]:
        pb = Pseudobulk(name, K["datasets"][name], K)
        rows, X = pb.aggregate(K["map_to_stage13"])
        loc, _ = S18.localisation(pb, rows, X, genes)
        locs.append(loc)
        for comp, M in metas.items():
            t = M.merge(loc, on="gene")
            t = t[t.expressed]
            t["cls"] = np.where(t.is_core, np.where(t.g_re > 0, "up", "down"), "none")
            for dr in ("up", "down"):
                for ct in sorted(t.top_type.unique()):
                    a = ((t.cls == dr) & (t.top_type == ct)).sum()
                    b = ((t.cls == dr) & (t.top_type != ct)).sum()
                    c = ((t.cls != dr) & (t.top_type == ct)).sum()
                    d = ((t.cls != dr) & (t.top_type != ct)).sum()
                    orr, p = stats.fisher_exact([[a, b], [c, d]])
                    enr.append({"dataset": name, "compartment": comp, "direction": dr, "top_type": ct, "n_core": int(a),
                                "frac_of_core": a / max(a + b, 1), "frac_background": (a + c) / len(t),
                                "odds_ratio": orr, "p": p})
    loc = pd.concat(locs)
    enr = pd.DataFrame(enr)
    enr["fdr"] = bh(enr.p.to_numpy())
    L = core.merge(loc, on="gene", how="left")
    return L, enr, loc


def marker_z(rng) -> pd.DataFrame:
    mk = pd.read_csv(ROOT / "results/14_kpmp/markers_kpmp.tsv", sep="\t")
    out = []
    for comp in SP["sources"]:
        M = pd.read_csv(OUT / "core" / f"meta_{comp}.tsv.gz", sep="\t").set_index("gene").g_re
        for (ref, ct), g in mk.groupby(["reference", "cell_type"]):
            gg = M.index.intersection(g.gene)
            if len(gg) < 10:
                continue
            obs = M[gg].mean()
            null = np.array([M.to_numpy()[rng.choice(len(M), len(gg), replace=False)].mean()
                             for _ in range(SP["kpmp"]["n_perm"])])
            out.append({"compartment": comp, "reference": ref, "cell_type": ct, "n_markers": len(gg),
                        "mean_g_re": obs, "z": (obs - null.mean()) / null.std(),
                        "frac_markers_core_up": float(np.isin(gg, _core(comp, "up")).mean()),
                        "frac_markers_core_down": float(np.isin(gg, _core(comp, "down")).mean())})
    return pd.DataFrame(out)


def _core(comp, dr):
    c = load_core_table()
    return c.gene[(c.compartment == comp) & (c.direction == dr)].to_numpy()


def cv_resid(X: np.ndarray, y: np.ndarray, reps: int, seed: int) -> tuple[float, np.ndarray]:
    r2, res = [], []
    for r in range(reps):
        pred = np.empty_like(y)
        for tr, te in KFold(CO["cv_folds"], shuffle=True, random_state=seed + r).split(X):
            pred[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
        r2.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        res.append(y - pred)
    return float(np.mean(r2)), np.mean(res, 0)


def composition(s, units, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    T = pd.read_csv(ROOT / "results/16_deconv/run" / CO["run_tag"] / "theta_type.tsv", sep="\t")
    r2rows, rho = [], []
    for comp, us in CO["units"].items():
        for u in us:
            idx = units[u]
            R = full_ranks(idx, s).dropna(axis=1)
            up, dn = loo_sets(comp, u, s, units)
            sc = signed_score(R, up, dn)
            d = s.loc[R.index, "diagnosis"].to_numpy()
            pat = patient_mask(d, include_dkd=True)
            ctrl = d == "CONTROL"
            for ref in CO["references"]:
                W = T[T.reference == ref].pivot_table(index="sample_uid", columns="type", values="theta").reindex(R.index)
                W = W.dropna(axis=1, how="all")
                ok = W.notna().all(1).to_numpy()
                if ok.sum() < CO["min_unit_n"]:
                    continue
                P = W[ok].loc[:, W[ok].std() > 0]
                C = clr(P, CO["eps"]).to_numpy()
                y = sc.to_numpy()[ok]
                r2_in = LinearRegression().fit(C, y).score(C, y)
                r2_cv, res = cv_resid(C, y, CO["cv_reps"], CFG["seed"])
                null = []
                for _ in range(50):
                    pick = rng.choice(R.shape[1], len(up) + len(dn), replace=False)
                    yr = (R.iloc[:, pick[:len(up)]].mean(1) - R.iloc[:, pick[len(up):]].mean(1)).to_numpy()[ok]
                    null.append(cv_resid(C, yr, 2, CFG["seed"])[0])
                row = {"reference": ref, "compartment": comp, "unit": u, "n": int(ok.sum()), "p": C.shape[1],
                       "n_controls": int(ctrl[ok].sum()), "r2": r2_in, "r2_cv": r2_cv,
                       "r2_cv_random_median": float(np.median(null)), "r2_cv_random_q95": float(np.quantile(null, 0.95))}
                pk = pat[ok]
                if pk.sum() >= CO["min_unit_n"]:
                    row["r2_cv_patients"] = cv_resid(C[pk], y[pk], CO["cv_reps"], CFG["seed"])[0]
                ck = ctrl[ok]
                if ck.sum() >= 3 and pk.sum() >= 3:
                    m = pk | ck
                    yy = pk[m].astype(int)
                    row |= {"auroc_raw": float(auc_rows(yy, y[m][None])[0]),
                            "auroc_composition_adjusted": float(auc_rows(yy, res[m][None])[0])}
                r2rows.append(row)
                for ct in P.columns:
                    lp = np.log(P[ct].clip(lower=CO["eps"]).to_numpy())
                    ra = stats.spearmanr(lp, y)
                    rp = stats.spearmanr(lp[pk], y[pk]) if pk.sum() >= 10 else (np.nan, np.nan)
                    rho.append({"reference": ref, "compartment": comp, "unit": u, "cell_type": ct,
                                "mean_theta": float(P[ct].mean()), "rho_all": ra[0], "p_all": ra[1],
                                "n_patients": int(pk.sum()), "rho_patients": rp[0], "p_patients": rp[1]})
    return pd.DataFrame(r2rows), pd.DataFrame(rho)


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    s, units = import_module("11_injury_axis").load_samples()
    core = load_core_table()
    L, enr, loc = localise(core)
    L.to_csv(O / "core_localisation.tsv", sep="\t", index=False)
    loc.to_csv(O / "kpmp_localisation_all_genes.tsv.gz", sep="\t", index=False)
    enr.to_csv(O / "core_top_type_enrichment.tsv", sep="\t", index=False)
    mz = marker_z(rng)
    mz.to_csv(O / "marker_program_z.tsv", sep="\t", index=False)
    r2, rho = composition(s, units, rng)
    r2.to_csv(O / "composition_r2.tsv", sep="\t", index=False)
    rho.to_csv(O / "composition_celltype_rho.tsv", sep="\t", index=False)
    s16 = pd.read_csv(ROOT / "results/16_deconv/axis_r2_summary.tsv", sep="\t")
    s16.to_csv(O / "stage16_axis_r2_summary.tsv", sep="\t", index=False)
    outs = [O / f for f in ("core_localisation.tsv", "core_top_type_enrichment.tsv", "marker_program_z.tsv",
                            "composition_r2.tsv", "composition_celltype_rho.tsv", "stage16_axis_r2_summary.tsv")]
    write_provenance("19_shared_program/cells", [OUT / "core/core_genes.tsv", ROOT / "results/14_kpmp/markers_kpmp.tsv",
                                                 ROOT / "results/16_deconv/run" / CO["run_tag"] / "theta_type.tsv"],
                     outs, CFG["seed"])
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        e = enr[(enr.fdr < 0.05) & (enr.odds_ratio > 1)]
        print(e.round(3).to_string())
        print(mz.pivot_table(index=["compartment", "cell_type"], columns="reference", values="z").round(1).to_string())
        print(r2.round(3).to_string())


if __name__ == "__main__":
    main()
