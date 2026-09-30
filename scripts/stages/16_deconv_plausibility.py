"""Stage 16c：反卷积比例的物理合理性 + 共享疾病–对照轴被组成解释的程度（均不读 DKD 标签、不含外部测试队列的诊断）。

1) proportions_by_batch.tsv  每个 (参照, 方法 BP/NNLS, series, 区室, 类型) 的中位数与四分位（全部样本 / 仅对照）。
2) key_checks.tsv            关键物理量：肾小球 bulk 中 PODO、肾小管污染（PT + DIST）、肾小球固有细胞内的 PODO 份额；
                             肾小管 bulk 中 PT、上皮总量、肾小球污染。
3) marker_concordance.tsv    θ_类型 与 bulk 中该类型 KPMP 标志（stage 14）平均秩的 Spearman（批内）。
4) axis_r2.tsv               单元内：共享轴分数 a（stage 15 同法、来源不含本单元）~ 组成（CLR）/ 上皮份额 / 组成 + 状态份额，
                             样本内 R²、校正 R²、重复 5 折交叉验证 R²；样本 = 非 DKD 样本（病人 + 对照）。
用法：python scripts/stages/16_deconv_plausibility.py [--tag all] [--reference snRNA]
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.deconv_feats import comp_features, statefrac_features  # noqa: E402
from lib.injury import full_ranks  # noqa: E402
from lib.m2_data import M2, _sources, _unit_score, unit_table  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
DC = CFG["deconv"]
PL = DC["plausibility"]
RUN = ROOT / "results" / "16_deconv" / "run"


def _q(x: pd.Series) -> pd.Series:
    return pd.Series({"median": x.median(), "q25": x.quantile(0.25), "q75": x.quantile(0.75), "n": len(x)})


def proportions(T: pd.DataFrame, N: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    D = pd.concat([T.assign(method="BayesPrism"), N.assign(method="NNLS")])
    D["diagnosis"] = s.diagnosis.reindex(D.sample_uid).values
    out = []
    for subset, d in (("all", D), ("controls", D[D.diagnosis == "CONTROL"])):
        g = d.groupby(["reference", "method", "series", "compartment", "type"]).theta.apply(_q).unstack()
        out.append(g.reset_index().assign(subset=subset))
    return pd.concat(out)


def key_checks(T: pd.DataFrame, N: pd.DataFrame, S: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for meth, D in (("BayesPrism", T), ("NNLS", N)):
        W = D.pivot_table(index=["reference", "series", "compartment", "sample_uid"], columns="type", values="theta")
        for (ref, ser, comp), w in W.groupby(level=[0, 1, 2]):
            w = w.fillna(0.0)
            ctrl = s.diagnosis.reindex(w.index.get_level_values(3)).values == "CONTROL"
            if DC["ref_compartment"][comp] == "GLOM":
                q = {"PODO": w.PODO, "tubular_contam(PT+DIST)": w.PT + w.DIST, "PT": w.PT,
                     "EC(all)": w.EC, "MES_VSMC": w.MES_VSMC, "PEC": w.PEC,
                     "immune": w.MYE + w.TNK + w.BPL,
                     "PODO_share_of_glom_intrinsic": w.PODO / (w.PODO + w.EC + w.MES_VSMC + w.PEC + 1e-12)}
            else:
                epi = [c for c in PL["tub_epithelial"] if c in w]
                q = {"PT": w.PT, "epithelial": w[epi].sum(1), "distal(TAL..IC)": w[[c for c in epi if c != "PT"]].sum(1),
                     "GLOM_contam": w.GLOM, "FIB": w.FIB, "immune": w.MYE + w.TNK + w.BPL}
            for k, v in q.items():
                for sub, m in (("all", np.ones(len(v), bool)), ("controls", ctrl)):
                    if m.sum():
                        rows.append({"reference": ref, "method": meth, "series": ser, "compartment": comp,
                                     "quantity": k, "subset": sub, "n": int(m.sum()),
                                     "median": float(np.median(v[m])), "q25": float(np.quantile(v[m], .25)),
                                     "q75": float(np.quantile(v[m], .75))})
    if not S.empty:                                    # 肾小球内皮（EC-GC 系列状态）占肾小球 bulk
        gc = S[S.state.isin(PL["ec_gc_states"])].groupby(["reference", "series", "compartment", "sample_uid"]).theta.sum()
        for (ref, ser, comp), v in gc.groupby(level=[0, 1, 2]):
            if DC["ref_compartment"][comp] == "GLOM":
                rows.append({"reference": ref, "method": "BayesPrism", "series": ser, "compartment": comp,
                             "quantity": "EC-GC_states(first)", "subset": "all", "n": len(v),
                             "median": float(v.median()), "q25": float(v.quantile(.25)), "q75": float(v.quantile(.75))})
    return pd.DataFrame(rows)


def marker_concordance(T: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    mk = pd.read_csv(ROOT / PL["marker_file"], sep="\t")
    mk = mk[mk.reference == PL["marker_reference"]]
    rows = []
    for (ref, ser, comp), d in T.groupby(["reference", "series", "compartment"]):
        W = d.pivot_table(index="sample_uid", columns="type", values="theta")
        R = full_ranks(W.index, s)
        for kc, tt in PL["marker_map"].items():
            t = tt[DC["ref_compartment"][comp]] if isinstance(tt, dict) else tt
            if t is None or t not in W:
                continue
            g = R.columns.intersection(mk.gene[mk.cell_type == kc])
            if len(g) < 10:
                continue
            sc = R[g].mean(1)
            rho = spearmanr(W[t], sc.reindex(W.index)).statistic
            rows.append({"reference": ref, "series": ser, "compartment": comp, "type": t, "kpmp_marker_set": kc,
                         "n_markers": len(g), "n": len(W), "spearman": rho})
    return pd.DataFrame(rows)


def _samples() -> pd.DataFrame:
    return import_module("11_injury_axis").load_samples()[0]


def _cv_r2(X: np.ndarray, y: np.ndarray, reps: int, seed: int) -> float:
    vals = []
    for r in range(reps):
        pred = np.empty_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=seed + r).split(X):
            pred[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
        vals.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(vals))


def axis_r2(ref: str, tag: str) -> pd.DataFrame:
    rows = []
    for comp in ("GLOM", "TUB"):
        cr = DC["ref_compartment"][comp]
        C, SF = comp_features(ref, cr, tag), statefrac_features(ref, cr, tag)
        t = unit_table(comp)
        W = pd.read_csv(RUN / tag / "theta_type.tsv", sep="\t").query("reference == @ref")
        W = W.pivot_table(index="sample_uid", columns="type", values="theta")
        epi_types = PL["glom_epithelial"] if cr == "GLOM" else PL["tub_epithelial"]
        for u in M2["units"][comp]:
            idx = t.index[(t.unit == u) & (t.diagnosis != "DKD")]
            a = _unit_score(comp, tuple(_sources(comp, {u})), u).reindex(idx)
            ok = a.notna() & C.reindex(idx).notna().all(1)
            idx, y = idx[ok.values], a[ok].to_numpy()
            if len(idx) < PL["min_unit_n"]:
                continue
            epi = W.reindex(idx)[epi_types].sum(1).clip(1e-3, 1 - 1e-3)
            sets = {"epithelial_logit": np.log(epi / (1 - epi)).to_numpy()[:, None],
                    "COMP": C.reindex(idx).to_numpy(),
                    "COMP+SF": np.c_[C.reindex(idx).to_numpy(), SF.reindex(idx).fillna(0).to_numpy()]}
            p_c = sets["COMP"].shape[1]                      # 对照：同维度的无监督 bulk 主成分 / 随机基因集得分
            R = full_ranks(idx, _samples()).dropna(axis=1)
            Rc = R.to_numpy() - R.to_numpy().mean(0)
            U, S_, _ = np.linalg.svd(Rc, full_matrices=False)
            sets[f"bulkPC{p_c}"] = U[:, :p_c] * S_[:p_c]
            rng = np.random.default_rng(CFG["seed"])
            sets[f"random{p_c}x{PL['random_set_size']}"] = np.column_stack(
                [R.iloc[:, rng.choice(R.shape[1], PL["random_set_size"], replace=False)].mean(1) for _ in range(p_c)])
            for nm, X in sets.items():
                X = X[:, X.std(0) > 1e-8]
                lr = LinearRegression().fit(X, y)
                r2 = lr.score(X, y)
                n, p = X.shape
                rows.append({"reference": ref, "compartment": comp, "unit": u, "features": nm, "n": n, "p": p,
                             "n_controls": int((t.loc[idx].diagnosis == "CONTROL").sum()), "r2": r2,
                             "r2_adj": 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1),
                             "r2_cv": _cv_r2(X, y, PL["cv_reps"], CFG["seed"]) if n >= 3 * p else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DC["eval"]["run_tag"])
    ap.add_argument("--reference", default=DC["primary"])
    args = ap.parse_args()
    set_global_seed(CFG["seed"])
    out = ROOT / "results" / "16_deconv" / "plausibility" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    s, _ = import_module("11_injury_axis").load_samples()
    T = pd.read_csv(RUN / args.tag / "theta_type.tsv", sep="\t")
    S = pd.read_csv(RUN / args.tag / "theta_state.tsv", sep="\t")
    N = pd.read_csv(RUN / args.tag / "theta_nnls.tsv", sep="\t")
    res = {"proportions_by_batch.tsv": proportions(T, N, s), "key_checks.tsv": key_checks(T, N, S, s),
           "marker_concordance.tsv": marker_concordance(T, s)}
    if args.tag == DC["eval"]["run_tag"]:
        res["axis_r2.tsv"] = pd.concat([axis_r2(r, args.tag) for r in sorted(T.reference.unique())])
    for f, df in res.items():
        df.to_csv(out / f, sep="\t", index=False)
    pd.set_option("display.width", 250)
    k = res["key_checks.tsv"]
    print(k[k.subset == "all"].pivot_table(index=["compartment", "quantity"], columns=["reference", "method"],
                                             values="median", aggfunc="median").round(3).to_string())
    if "axis_r2.tsv" in res:
        print(res["axis_r2.tsv"].round(3).to_string(index=False))
    write_provenance(f"16_deconv/plausibility/{args.tag}", [RUN / args.tag / f for f in
                     ("theta_type.tsv", "theta_state.tsv", "theta_nnls.tsv")], [out / f for f in res], CFG["seed"],
                     {"tag": args.tag})


if __name__ == "__main__":
    main()
