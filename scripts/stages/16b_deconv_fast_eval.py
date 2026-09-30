"""Stage 16b-2：组成 / 细胞内状态特征的非 DKD dev 评估（与 stage 15 同 fold、同 cell、同共享轴协变量）。

只用非 DKD dev 任务（E1 留一单元、E2 跨联盟、E4 耦合模拟）；DKD 不进任何测试集（assert_no_dkd_eval）。
DKD vs 其他 CKD 的 ERCB H1↔H7 最终评估在本脚本中不存在。
同一 L2 logistic（lib.evalkit.fit_logistic_cv：标准化 + inner LOCO 选 C + class_weight balanced），特征集：
  TISSUE（= B-L2，读取 stage 15 r5 的缓存预测，逐 cell 同一定义）/ COMP（MLP log 比例）/ COMP_NNLS /
  STATE_A / STATE_R / COMP+STATE_A / COMP+STATE_R（类型级 K 维状态）；
  TISSUE_G（16b 基因集上的组织秩）/ STATE_RG（全部基因的组成回归残差）/ STATE_AG（每类型特异基因块上的分配残差）
  及其 COMP+ 组合（基因级状态）；TISSUE+COMP+STATE_R（可选 --with-tissue-combo）。
输出 results/16b_deconv_fast/eval/：preds/、cell_metrics.tsv、summary.tsv、paired_vs_BL2.tsv、axis_r2.tsv。
用法：python scripts/stages/16b_deconv_fast_eval.py [--n-jobs 48] [--with-tissue-combo]
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.deconv_fast_ref import build_compref, donor_split, load_pseudobulk, markers, prior_mean  # noqa: E402
from lib.deconv_fast_state import logp, resid_genes, state_alloc_genes, state_resid  # noqa: E402
from lib.evalkit import fit_logistic_cv  # noqa: E402
from lib.m2_data import covariate, features, impute, make_folds  # noqa: E402
from lib.m2_eval import assert_no_dkd_eval, metrics_from_preds  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
DF = CFG["deconv_fast"]
EV = DF["eval"]
IN = ROOT / "results" / "16b_deconv_fast"
OUT = IN / "eval"
D15 = import_module("15_dev")


def build_stateAG(comp: str) -> Path:
    """基因级分配状态（样本级、不在 bulk 上拟合）：按 stage 16b-1 同一供体划分 / 基因集重建签名，缓存 parquet。"""
    p = ROOT / "data/interim" / f"16b_stateAG_{comp}.parquet"
    R = pd.read_parquet(ROOT / "data/interim" / f"16b_ranks_{comp}.parquet")
    P = pd.read_csv(IN / f"props_{comp}_mlp.tsv", sep="\t", index_col=0)
    pb = load_pseudobulk()
    donors = pd.read_csv(IN / "donor_split.tsv", sep="\t", index_col=0)
    assert donors.equals(donor_split(pb, CFG["seed"])), "供体划分与 16b-1 不一致"
    ref = build_compref(pb, comp, donors, R.columns).subset_genes(R.columns)
    S = ref.signature(np.flatnonzero(donors.split.values == "train"))
    mix = S @ prior_mean(ref)
    ref_log = np.log2(mix / mix.sum() * 1e6 + 1)
    blocks = markers(S, ref.genes, ref.types, DF["state"]["n_block"])
    X = state_alloc_genes(R.to_numpy(), P[ref.types].to_numpy(), S, blocks, ref.genes, ref.types, ref_log)
    cols = [f"{t}|{g}" for t in ref.types for g in blocks[t]]
    pd.DataFrame(X, index=R.index, columns=cols).astype("float32").to_parquet(p)
    return p


def load_comp(comp: str) -> dict:
    rd = lambda f: pd.read_csv(IN / f, sep="\t", index_col=0)  # noqa: E731
    R = pd.read_parquet(ROOT / "data/interim" / f"16b_ranks_{comp}.parquet")
    mk = pd.read_csv(IN / f"markers_{comp}.tsv", sep="\t")
    P = rd(f"props_{comp}_mlp.tsv")
    mk_idx = [R.columns.get_indexer(mk.gene[mk.cell_type == t]) for t in P.columns]
    AG = pd.read_parquet(ROOT / "data/interim" / f"16b_stateAG_{comp}.parquet")
    return {"P": P, "Pn": rd(f"props_{comp}_nnls.tsv"), "A": rd(f"stateA_{comp}.tsv"), "R": R, "mk_idx": mk_idx,
            "AG": AG}


def fold_features(fold, D: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    tr, te = fold.train.index, fold.test.index
    P, Pn, A, R = D["P"], D["Pn"], D["A"], D["R"]
    Str, Ste = state_resid(R.loc[tr].to_numpy(), R.loc[te].to_numpy(), P.loc[tr].to_numpy(), P.loc[te].to_numpy(),
                           D["mk_idx"])
    c = lambda M: (logp(M.loc[tr].to_numpy()), logp(M.loc[te].to_numpy()))  # noqa: E731
    comp, compn = c(P), c(Pn)
    sa = (A.loc[tr].to_numpy(), A.loc[te].to_numpy())
    sr = (Str, Ste)
    cat = lambda a, b: (np.c_[a[0], b[0]], np.c_[a[1], b[1]])  # noqa: E731
    rg = resid_genes(R.loc[tr].to_numpy(), R.loc[te].to_numpy(), P.loc[tr].to_numpy(), P.loc[te].to_numpy())
    ag = (D["AG"].loc[tr].to_numpy(), D["AG"].loc[te].to_numpy())
    return {"COMP": comp, "COMP_NNLS": compn, "STATE_A": sa, "STATE_R": sr,
            "COMP+STATE_A": cat(comp, sa), "COMP+STATE_R": cat(comp, sr),
            "TISSUE_G": (R.loc[tr].to_numpy(), R.loc[te].to_numpy()),  # 同一 16b 基因集上的组织平均：分离基因集效应
            "STATE_RG": rg, "COMP+STATE_RG": cat(comp, rg), "STATE_AG": ag, "COMP+STATE_AG": cat(comp, ag)}


def _fit_one(Xtr, Xte, train, d, seed):
    pat = train.role.isin(["head", "neg"]).to_numpy()
    y = (train.diagnosis.to_numpy() == d).astype(int)
    f = fit_logistic_cv(Xtr[pat], y[pat], train.unit.to_numpy()[pat], CFG["probe"]["C_grid"], "l2", CFG, seed)
    return d, f["model"].decision_function(Xte)


def tissue_combo(fold, F: pd.DataFrame, feats: dict) -> tuple[np.ndarray, np.ndarray]:
    """TISSUE+COMP+STATE_R：B-L2 同一基因选择（训练病人方差 top baseline_max_genes）后拼接低维特征。"""
    Xtr, Xte, _ = impute(F.loc[fold.train.index].to_numpy(np.float32), F.loc[fold.test.index].to_numpy(np.float32))
    pat = fold.train.role.isin(["head", "neg"]).to_numpy()
    g = np.argsort(-Xtr[pat].var(0))[:CFG["model2"]["baseline_max_genes"]]
    a, b = feats["COMP+STATE_R"]
    return np.c_[Xtr[:, g], a], np.c_[Xte[:, g], b]


def fold_job(fam: str, comp: str, i: int, with_combo: bool) -> pd.DataFrame:
    folds = make_folds(fam, comp)
    fold = folds[i]
    cells = D15.fold_cells(folds, fam)[fold.key]
    if not cells:
        return pd.DataFrame()
    assert_no_dkd_eval(fold.test)
    D = load_comp(comp)
    feats = fold_features(fold, D)
    if with_combo:
        feats["TISSUE+COMP+STATE_R"] = tissue_combo(fold, features(), feats)
    a = covariate(fold)
    out = []
    for name, (Xtr, Xte) in feats.items():
        sc = dict(_fit_one(Xtr, Xte, fold.train, d, CFG["seed"]) for d in cells)
        out.append(D15.long_rows(fold, cells, sc, a, name))
    return pd.concat(out)


def axis_r2(fam_units: list[tuple[str, int]]) -> pd.DataFrame:
    """E1 每个测试单元：共享轴分数 a（该 fold 的来源，不含测试单元）被组成 / 状态解释的 R²（样本内与 5 折 CV）。"""
    rows = []
    for comp, i in fam_units:
        fold = make_folds("E1", comp)[i]
        D = load_comp(comp)
        te = fold.test.index                                    # 测试单元全部非 DKD 样本（含对照）
        a = covariate(fold).loc[te].to_numpy()
        feats = fold_features(fold, D)
        for name in ["COMP", "COMP_NNLS", "STATE_A", "STATE_R", "COMP+STATE_A", "COMP+STATE_R"]:
            X = feats[name][1]
            if len(te) < 3 * X.shape[1]:
                continue
            ins = LinearRegression().fit(X, a).score(X, a)
            cv = cross_val_predict(LinearRegression(), X, a,
                                   cv=KFold(EV["r2_cv_folds"], shuffle=True, random_state=CFG["seed"]))
            rows.append({"compartment": comp, "unit": fold.test_unit, "features": name, "n": len(te),
                         "p": X.shape[1], "r2_in": ins, "r2_cv": 1 - ((a - cv) ** 2).sum() / ((a - a.mean()) ** 2).sum()})
    return pd.DataFrame(rows)


def paired(m: pd.DataFrame, ref: str) -> pd.DataFrame:
    key = ["family", "compartment", "test_unit", "disease"]
    base = m[m.method == ref].set_index(key)
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    for meth in sorted(set(m.method) - {ref}):
        j = m[m.method == meth].set_index(key).join(base, rsuffix="_ref", how="inner")
        fam = j.index.get_level_values(0)
        unit = pd.Index(j.index.get_level_values(2))
        groups = {f: fam == f for f in sorted(set(fam))} | {"ALL": np.ones(len(j), bool),
                                                             "ERCB": unit.str.startswith("ERCB_")}
        for g, mask in groups.items():
            jj = j[np.asarray(mask)]
            r = {"method": meth, "family": g, "n_cells": len(jj)}
            for k in ["auroc", "adj_auroc"]:
                dlt = (jj[k] - jj[f"{k}_ref"]).dropna().to_numpy()
                boot = [rng.choice(dlt, len(dlt)).mean() for _ in range(EV["n_boot"])]
                r |= {f"{k}": jj[k].mean(), f"{k}_BL2": jj[f"{k}_ref"].mean(), f"d_{k}": dlt.mean(),
                      f"d_{k}_lo": np.quantile(boot, 0.025), f"d_{k}_hi": np.quantile(boot, 0.975),
                      f"d_{k}_winrate": (dlt > 0).mean(),
                      f"d_{k}_wilcoxon_p": wilcoxon(dlt).pvalue if len(dlt) >= 6 and (dlt != 0).any() else np.nan}
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=48)
    ap.add_argument("--with-tissue-combo", action="store_true")
    args = ap.parse_args()
    set_global_seed(CFG["seed"])
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    for comp in ["GLOM", "TUB"]:
        build_stateAG(comp)
    jobs = [(fam, comp, i) for fam in EV["families"] for comp in ["GLOM", "TUB"]
            for i in range(len(make_folds(fam, comp)))]
    res = Parallel(n_jobs=args.n_jobs)(delayed(fold_job)(*j, args.with_tissue_combo) for j in jobs)
    P = pd.concat([r for r in res if len(r)])
    P.to_csv(OUT / "preds" / "deconv_fast_preds.tsv.gz", sep="\t", index=False)
    bdir = ROOT / "results/15_model" / EV["baseline_run"] / "preds" / EV["baseline"]
    B = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(bdir.glob("*.tsv"))])
    B = B[B.family.isin(EV["families"])].assign(method="TISSUE(B-L2)")
    m = metrics_from_preds(pd.concat([P, B[P.columns]]))
    m.to_csv(OUT / "cell_metrics.tsv", sep="\t", index=False)
    n_cells = m.groupby("method").size()
    assert n_cells.nunique() == 1, f"cell 数不一致：{n_cells.to_dict()}"
    s = m.groupby(["method"])[["auroc", "adj_auroc", "auroc_ws"]].mean().assign(n_cells=n_cells)
    s.to_csv(OUT / "summary.tsv", sep="\t")
    pr = paired(m, "TISSUE(B-L2)")
    pr.to_csv(OUT / "paired_vs_BL2.tsv", sep="\t", index=False)
    e1 = [(c, i) for c in ["GLOM", "TUB"] for i, f in enumerate(make_folds("E1", c))]
    r2 = axis_r2(e1)
    r2.to_csv(OUT / "axis_r2.tsv", sep="\t", index=False)
    print(s.round(4).to_string())
    print(pr[pr.family.isin(["ALL", "E1", "E2", "E4"])].round(4).to_string())
    print(r2.round(3).to_string())
    write_provenance("16b_deconv_fast/eval", sorted(IN.glob("*.tsv")) + sorted(bdir.glob("*.tsv")),
                     [OUT / f for f in ["cell_metrics.tsv", "summary.tsv", "paired_vs_BL2.tsv", "axis_r2.tsv"]],
                     CFG["seed"], {"families": EV["families"], "n_cells": int(n_cells.iloc[0]),
                                   "with_tissue_combo": args.with_tissue_combo})


if __name__ == "__main__":
    main()
