"""Stage 18c：DKD 关键基因的 bulk 验证链（独立留出批次、与已发表签名同口径审计、GSE142025 进展）。

A 独立性（无循环）：ERCB 四个测试单元 u，基因选择只用「不含 u 训练」的 E1_u 模型（DKD head 标准化总效应 e 的
  top-k，带模型符号）。在 u 的病人（与 final_dkd 同一测试集）中逐基因拟合 rank ~ DKD + a（a = final_dkd 预测表中
  的 fold 共享轴分数），报告 DKD 系数 / 残差 SD（标准化效应）、P、方向是否与模型一致；集合层面报告方向一致率
  （DKD 标签置换 P）。FULL 关键基因另报：在哪些 u 中被 E1_u 模型选中（top_k），并只在这些 u 上做 Stouffer 合并。
B 已发表签名审计：stage 12 的 metrics / score_sig / random_null（同一损伤分数、同一随机集定向方式），评估同一基因集，
  与 stage 12 的已发表签名表对照（调整 AUROC 百分位 ≥ 0.95 的比例）。
C 进展：GSE142025（从未进入任何训练）早期 vs 晚期 DKD，FULL TUB 模型 top-k 基因评分（样本内秩），原始与共享轴调整。
输出 results/18_interpret/validate/。
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.injury import adjusted_auroc, full_ranks  # noqa: E402
from lib.interp import signed_set_score  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
IC = CFG["interpret"]
W = ROOT / "results/18_interpret/weights"
OUT = ROOT / "results/18_interpret/validate"
ERCB = {"GLOM": ["ERCB_GLOM_H1", "ERCB_GLOM_H7"], "TUB": ["ERCB_TUB_H1", "ERCB_TUB_H7"]}
S12 = import_module("12_signature_audit")


def fold_effects(comp: str, key: str) -> pd.Series:
    z = np.load(ROOT / IC["models_dir"] / f"{key}.npz")
    h = list(z["heads"]).index("DKD")
    from lib.m2_data import universe
    e = ((z["gp"][:, :, h] + z["gr"][:, :, h]) * z["x_sd"]).mean(0)
    return pd.Series(e, index=universe())


def top_signed(e: pd.Series, k: int) -> pd.Series:
    t = e.abs().nlargest(k).index
    return np.sign(e[t]).astype(int)


def gene_tests(R: pd.DataFrame, y: np.ndarray, a: np.ndarray) -> pd.DataFrame:
    X = sm.add_constant(np.c_[y, a])
    rows = []
    for g in R.columns:
        v = R[g].to_numpy(float)
        f = sm.OLS(v, X).fit()
        rows.append({"gene": g, "coef": f.params[1], "p": f.pvalues[1], "d_adj": f.params[1] / np.sqrt(f.scale),
                     "t": f.tvalues[1], "auroc_raw": roc_auc_score(y, v)})
    return pd.DataFrame(rows).set_index("gene")


def concordance_perm(R: pd.DataFrame, sign: pd.Series, y: np.ndarray, a: np.ndarray, rng, n_perm: int) -> tuple:
    """集合层面：方向一致率（DKD 系数符号 = 模型符号）与签名调整 AUROC；DKD 标签在单元内置换。"""
    X0 = R[sign.index].to_numpy(float)

    def stat(yy):
        D = sm.add_constant(np.c_[yy, a])
        B = np.linalg.lstsq(D, X0, rcond=None)[0][1]
        conc = float((np.sign(B) == sign.to_numpy()).mean())
        sc = X0 @ sign.to_numpy() / len(sign)
        return conc, adjusted_auroc(yy, sc, a)
    obs = stat(y)
    null = np.array([stat(rng.permutation(y)) for _ in range(n_perm)])
    return obs, float((1 + (null[:, 0] >= obs[0]).sum()) / (1 + n_perm)), \
        float((1 + (null[:, 1] >= obs[1]).sum()) / (1 + n_perm)), float(null[:, 0].mean())


def part_a(s, rng):
    P = pd.read_csv(ROOT / "results/15_model/final_dkd/predictions.tsv", sep="\t")
    P = P[P.method == "RRG-ID"]
    key = pd.read_csv(W / "key_genes.tsv", sep="\t")
    key = key[key["head"] == "DKD"]
    per_gene, sets, keyrows = [], [], []
    for comp, units in ERCB.items():
        for u in units:
            te = P[P.test_unit == u].set_index("sample_uid")
            R = full_ranks(te.index, s)
            y, a = te.y.to_numpy(), te.a.to_numpy(float)
            e = fold_effects(comp, f"E1_{comp}_{u}")
            e = e[e.index.isin(R.columns[R.notna().all()])]
            T = gene_tests(R[top_signed(e, max(IC["sig_sizes"] + [IC["top_k"]])).index], y, a)
            for k in IC["sig_sizes"]:
                sg = top_signed(e, k)
                t = T.loc[sg.index]
                (conc, adj), p_conc, p_adj, null_conc = concordance_perm(R, sg, y, a, rng, IC["n_perm"])
                sets.append({"compartment": comp, "test_unit": u, "k": k, "n_dkd": int(y.sum()), "n_other": int((1 - y).sum()),
                             "concordance": conc, "concordance_null_mean": null_conc, "p_concordance_perm": p_conc,
                             "adj_auroc_set": adj, "p_adj_auroc_perm": p_adj,
                             "n_nominal_p05_concordant": int(((t.p < .05) & (np.sign(t.coef) == sg)).sum()),
                             "n_nominal_p05_discordant": int(((t.p < .05) & (np.sign(t.coef) != sg)).sum()),
                             "median_signed_d_adj": float((t.d_adj * sg).median())})
            topk = top_signed(e, IC["top_k"])
            T2 = gene_tests(R[[g for g in key[key.compartment == comp].gene if g in R.columns]], y, a)
            for g in key[key.compartment == comp].gene:
                if g not in T2.index:
                    continue
                sel = g in topk.index
                sg_full = np.sign(key.set_index(["compartment", "gene"]).e_full[(comp, g)])
                keyrows.append({"compartment": comp, "test_unit": u, "gene": g, "selected_without_unit": sel,
                                "sign_model": int(sg_full), "sign_fold_model": int(topk.get(g, 0)),
                                **T2.loc[g].to_dict()})
            per_gene.append(T.assign(compartment=comp, test_unit=u, rank_in_fold=np.arange(1, len(T) + 1),
                                     sign_model=top_signed(e, len(T)).reindex(T.index).values).reset_index())
            print(u, "A done", flush=True)
    K = pd.DataFrame(keyrows)
    K["z_signed"] = np.sign(K.coef) * stats.norm.isf(K.p / 2) * K.sign_model
    meta = []
    for (comp, g), d in K.groupby(["compartment", "gene"]):
        ok = d[d.selected_without_unit]
        zz = ok.z_signed.to_numpy()
        meta.append({"compartment": comp, "gene": g, "n_units_selected_without": len(ok),
                     "units": ",".join(ok.test_unit), "stouffer_z": zz.sum() / np.sqrt(len(zz)) if len(zz) else np.nan,
                     "p_one_sided": stats.norm.sf(zz.sum() / np.sqrt(len(zz))) if len(zz) else np.nan,
                     "n_concordant": int((zz > 0).sum()), "mean_signed_d_adj": float((ok.d_adj * ok.sign_model).mean()) if len(ok) else np.nan})
    return pd.concat(per_gene), pd.DataFrame(sets), K, pd.DataFrame(meta)


def part_b(s, units, rng):
    inj_all = pd.read_csv(ROOT / "results/11_injury/injury_scores.tsv", sep="\t")
    orient = S12.orientation(s, units)
    pub = pd.read_csv(ROOT / "results/12_signature_audit/signature_metrics.tsv", sep="\t")
    rows = []
    evals = [(c, u, f"E1_{c}_{u}") for c, us in ERCB.items() for u in us] + [("TUB", "GSE142025-RNAseq", "FULL_TUB")]
    for comp, ev, key in evals:
        ij = inj_all[(inj_all.eval_unit == ev) & (inj_all.compartment == comp)].set_index("sample_uid")
        R = full_ranks(ij.index, s).dropna(axis=1)
        diag, inj = ij.diagnosis.values, ij.injury.values
        e = fold_effects(comp, key)
        e = e[e.index.isin(R.columns)]
        for k in IC["sig_sizes"]:
            gd = top_signed(e, k).to_dict()
            rec = {"sig_id": f"RRG_DKD_top{k}", "compartment": comp, "eval_unit": ev, "gene_source_model": key,
                   "n_genes_used": len(gd), **S12.metrics(S12.score_sig(R, gd), diag, inj)}
            nl = S12.random_null(R, orient[comp], len(gd), diag, inj, rng)
            for m in ("auc_dkd_ctrl", "auc_dkd_other", "adj_auc_dkd_other", "specificity_ratio"):
                v = rec[m]
                rec[f"{m}_null_median"] = float(nl[m].median())
                rec[f"{m}_pct_vs_random"] = float((nl[m] < v).mean()) if not np.isnan(v) else np.nan
            p = pub[(pub.eval_unit == ev) & pub.sig_id.str.startswith("PMID")]
            rec["n_published"] = len(p)
            rec["n_published_adj_pct_ge95"] = int((p.adj_auc_dkd_other_pct_vs_random >= .95).sum())
            rec["frac_published_adj_pct_ge95"] = float((p.adj_auc_dkd_other_pct_vs_random >= .95).mean()) if len(p) else np.nan
            rec["rank_adj_auc_among_published"] = float((p.adj_auc_dkd_other.dropna() < rec["adj_auc_dkd_other"]).mean()) \
                if p.adj_auc_dkd_other.notna().any() else np.nan
            rec["rank_auc_ctrl_among_published"] = float((p.auc_dkd_ctrl.dropna() < rec["auc_dkd_ctrl"]).mean()) \
                if p.auc_dkd_ctrl.notna().any() else np.nan
            rows.append(rec)
        print(ev, "B done", flush=True)
    B = pd.DataFrame(rows)
    rep = []                                                 # 已发表签名：两个 ERCB 批次都 ≥0.95 的数量（复现）
    for comp, us in ERCB.items():
        p = pub[pub.eval_unit.isin(us) & pub.sig_id.str.startswith("PMID")]
        w = p.pivot_table(index="sig_id", columns="eval_unit", values="adj_auc_dkd_other_pct_vs_random")
        rep.append({"compartment": comp, "n_published": len(w), "n_ge95_any": int((w >= .95).any(axis=1).sum()),
                    "n_ge95_both": int((w >= .95).all(axis=1).sum())})
    return B, pd.DataFrame(rep)


def stage_labels(series: str) -> pd.Series:
    m = pd.read_csv(ROOT / "data/interim" / f"{series}_meta.tsv", sep="\t")
    lab = m.characteristics_ch1.str.extract(r"group:\s*(\S+)")[0]
    return pd.Series(lab.values, index=series + "|" + m.gsm)


def part_c(s, units, rng):
    g = IC["gse142025"]
    lab = stage_labels(g["series"])
    inj = pd.read_csv(ROOT / "results/11_injury/injury_scores.tsv", sep="\t")
    inj = inj[inj.eval_unit == g["series"]].set_index("sample_uid").injury
    uids = lab.index[lab.isin([g["early"], g["late"], "Control"])]
    R = full_ranks(pd.Index(uids), s).dropna(axis=1)
    lab, a = lab[R.index], inj.reindex(R.index)
    e = fold_effects("TUB", "FULL_TUB")
    e = e[e.index.isin(R.columns)]
    key = pd.read_csv(W / "key_genes.tsv", sep="\t")
    key = key[(key["head"] == "DKD") & (key.compartment == "TUB") & key.gene.isin(R.columns)]
    orient = S12.orientation(s, units)["TUB"]
    pool = R.columns.intersection(orient.index).values
    sets = {f"top{k}": top_signed(e, k) for k in IC["sig_sizes"]}
    sets["key_TUB"] = pd.Series(np.sign(key.e_full.values).astype(int), index=key.gene.values)
    rows, per = [], []
    for name, sg in sets.items():
        sc = pd.Series(signed_set_score(R, sg), index=R.index)
        lt = lab == g["late"]
        dn = lab.isin([g["early"], g["late"]])
        x_e, x_l = sc[lab == g["early"]], sc[lt]
        D = sm.add_constant(np.c_[lt[dn].astype(float), a[dn]])
        f = sm.OLS(sc[dn].to_numpy(), D).fit()
        null = []
        for _ in range(IC["n_random_sets"]):
            gg = rng.choice(pool, len(sg), replace=False)
            w = np.sign(orient[gg].values); w[w == 0] = 1
            v = R[gg].to_numpy() @ w / len(gg)
            null.append(roc_auc_score(lt[dn], v[dn.values]))
        auc = roc_auc_score(lt[dn], sc[dn])
        rows.append({"set": name, "n_genes": len(sg), "n_early": len(x_e), "n_late": len(x_l),
                     "n_control": int((lab == "Control").sum()),
                     "auroc_late_vs_early": auc, "hedges_g": S12.cohen_d(x_l.to_numpy(), x_e.to_numpy()),
                     "mwu_p": stats.mannwhitneyu(x_l, x_e).pvalue, "coef_late_adj_axis": f.params[1],
                     "p_late_adj_axis": f.pvalues[1], "auroc_null_median": float(np.median(null)),
                     "auroc_pct_vs_random": float((np.array(null) < auc).mean()),
                     "auroc_dkd_vs_control": roc_auc_score(dn, sc), "r_score_axis_dkd": stats.spearmanr(sc[dn], a[dn])[0]})
        per.append(pd.DataFrame({"set": name, "sample_uid": R.index, "group": lab.values, "score": sc.values, "a": a.values}))
    return pd.DataFrame(rows), pd.concat(per)


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(ROOT / "data/processed/samples.tsv", sep="\t").set_index("sample_uid")
    units = import_module("11_injury_axis").load_samples()[1]
    per_gene, sets, K, meta = part_a(s, rng)
    B, rep = part_b(s, units, rng)
    C, Cs = part_c(s, units, rng)
    files = {"heldout_gene_tests.tsv": per_gene, "heldout_set_tests.tsv": sets, "heldout_key_genes.tsv": K,
             "heldout_key_genes_meta.tsv": meta, "signature_audit.tsv": B, "published_replication.tsv": rep,
             "gse142025_progression.tsv": C, "gse142025_scores.tsv": Cs}
    for f, t in files.items():
        t.to_csv(OUT / f, sep="\t", index=False)
    write_provenance("18_interpret/validate", [W / "key_genes.tsv", ROOT / "results/15_model/final_dkd/predictions.tsv",
                                               ROOT / "results/11_injury/injury_scores.tsv",
                                               ROOT / "results/12_signature_audit/signature_metrics.tsv"],
                     [OUT / f for f in files], CFG["seed"], {"n_perm": IC["n_perm"], "sig_sizes": IC["sig_sizes"]})
    with pd.option_context("display.width", 250):
        print(sets.round(3).to_string()), print(meta.round(3).to_string())
        print(B.round(3).to_string()), print(rep.to_string()), print(C.round(3).to_string())


if __name__ == "__main__":
    main()
