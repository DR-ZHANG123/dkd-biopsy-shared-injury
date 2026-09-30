"""Stage 18d：DKD 关键基因的 KPMP 单细胞定位与供体水平方向（供体为重复单位）；eGFR 供体水平相关。

细胞类型口径 = stage 14 的 map_to_stage13（coarse）。对每个数据集（snRNA 主、scRNA 复现）：
  定位：健康参考供体（REF）× 细胞类型 pseudobulk log2CPM（n_cells ≥ min_cells），细胞类型均值；
        每基因报告最高表达细胞类型、跨类型特异性 z（同 stage 14 _spec_z）、tau 指数、最高类型份额。
  方向：stage 14 de_kpmp.tsv.gz（Welch，供体为单位）中该基因在其最高表达类型（及全部类型）里
        DKD vs REF / HKD / OTHER_CKD 的 log2FC、Hedges g、P、FDR，与模型符号比较。
  集合方向检验：stage 14 direction_test（供体标签置换）：关键基因 / top100（模型符号）在各细胞类型的方向一致率
        相对全宇宙背景的差（delta）与置换 P。
eGFR（仅 snRNA 有分箱 eGFR，无蛋白尿）：全组织 pseudobulk（供体全部细胞之和）log2CPM，关键基因在供体间 z 标准化后
  按模型符号平均为供体分数；DKD 供体内、全部 CKD 供体内与 eGFR 分箱中点的 Spearman；逐基因 Spearman（BH）。
输出 results/18_interpret/kpmp/。
"""
from __future__ import annotations

import re
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.kpmp import Pseudobulk  # noqa: E402
from lib.kpmp_stats import bh, direction_test, log2cpm  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
IC, K = CFG["interpret"], CFG["kpmp"]
W = ROOT / "results/18_interpret/weights"
OUT = ROOT / "results/18_interpret/kpmp"
S14 = import_module("14_kpmp_validation")
EGFR = "Baseline eGFR (ml/min/1.73m2) (Binned)"


def gene_sets() -> tuple[dict[str, pd.Series], pd.DataFrame]:
    G = pd.read_csv(W / "gene_effects.tsv.gz", sep="\t")
    G = G[G["head"] == "DKD"]
    sets = {}
    for comp, g in G.groupby("compartment"):
        g = g.set_index("gene")
        sets[f"key_{comp}"] = np.sign(g.e_full[g.is_key]).astype(int)
        sets[f"top100_{comp}"] = np.sign(g.e_full[g.e_full.abs().nlargest(100).index]).astype(int)
    return sets, G


def localisation(pb: Pseudobulk, rows, X, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = pb.group_donors("REF")
    ok = rows.donor.isin(ref) & (rows.n_cells >= K["min_cells"])
    gi = pb.genes.get_indexer(genes)
    L = log2cpm(X[np.flatnonzero(ok.values)])
    gl = [g for g, i in zip(genes, gi) if i >= 0]
    Lg = pd.DataFrame(L[:, gi[gi >= 0]], columns=gl)
    mean_ct = Lg.groupby(rows[ok].cell_type.values).mean().T          # 基因 × 类型
    z = mean_ct.sub(mean_ct.mean(1), axis=0).div(mean_ct.std(1) + 1e-6, axis=0)
    lin = (2 ** mean_ct - 1)
    tau = (1 - lin.div(lin.max(1) + 1e-12, axis=0)).sum(1) / (lin.shape[1] - 1)
    loc = pd.DataFrame({"dataset": pb.name, "gene": mean_ct.index, "top_type": mean_ct.idxmax(1).values,
                        "second_type": mean_ct.apply(lambda r: r.nlargest(2).index[-1], axis=1).values,
                        "top_log2cpm": mean_ct.max(1).values, "top_z": z.max(1).values, "tau": tau.values,
                        "top_share": (lin.max(1) / (lin.sum(1) + 1e-12)).values,
                        "expressed": (mean_ct.max(1) >= IC["spec_min_log2cpm"]).values})
    long = mean_ct.stack().rename("mean_log2cpm").reset_index().rename(columns={"level_0": "gene", "level_1": "cell_type"})
    long["spec_z"] = z.stack().values
    return loc, long.assign(dataset=pb.name)


def direction_rows(DE: pd.DataFrame, loc: pd.DataFrame, sets: dict, G: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, sg in sets.items():
        comp = name.split("_")[1]
        for g, sign in sg.items():
            l = loc[loc.gene == g]
            if l.empty:
                continue
            l = l.iloc[0]
            for con in IC["kpmp_contrasts"]:
                d = DE[(DE.contrast == con) & (DE.gene == g)].set_index("cell_type")
                for role, ct in (("top", l.top_type), ("second", l.second_type)):
                    r = d.loc[ct] if ct in d.index else None
                    out.append({"dataset": l.dataset, "set": name, "compartment": comp, "gene": g, "sign_model": sign,
                                "contrast": con, "role": role, "cell_type": ct,
                                **({k: r[k] for k in ("n_A", "n_B", "log2fc", "g", "p", "fdr")} if r is not None else {}),
                                "concordant": (np.sign(r.log2fc) == sign) if r is not None else np.nan})
    return pd.DataFrame(out)


def set_direction(pb, rows, X, sets, G, rng) -> pd.DataFrame:
    out = []
    for con in IC["kpmp_contrasts"]:
        a, b = con.split("_vs_")
        da, db = pb.group_donors(a), pb.group_donors(b)
        per = S14.expressed(rows, X, da.union(db))
        for ct, (r, L) in per.items():
            ia, ib = np.flatnonzero(r.donor.isin(da).values), np.flatnonzero(r.donor.isin(db).values)
            if len(ia) < K["min_donors"] or len(ib) < K["min_donors"]:
                continue
            keep = L[np.r_[ia, ib]].mean(0) >= K["min_expr_log2cpm"]
            genes = pb.genes[keep]
            Lab = np.vstack([L[ia][:, keep], L[ib][:, keep]])
            for name, sg in sets.items():
                comp = name.split("_")[1]
                bg = G[G.compartment == comp].set_index("gene").e_full
                t = direction_test(Lab, genes, len(ia), bg[sg.index], bg, K["lfc_min"], IC["n_perm"], rng)
                if t:
                    out.append({"dataset": pb.name, "set": name, "contrast": con, "cell_type": ct,
                                "n_A": len(ia), "n_B": len(ib)} | t)
    return pd.DataFrame(out)


def egfr_mid(v: str) -> float:
    m = re.match(r"(\d+)-(\d+)", str(v))
    return (int(m.group(1)) + int(m.group(2)) + 1) / 2 if m else np.nan


def _rank_resid(v: np.ndarray, c: np.ndarray) -> np.ndarray:
    rv, rc = stats.rankdata(v), stats.rankdata(c)
    X = np.c_[np.ones(len(rc)), rc]
    return rv - X @ np.linalg.lstsq(X, rv, rcond=None)[0]


def egfr_tests(pb: Pseudobulk, sets: dict, G: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """供体分数 ~ eGFR：Spearman；偏 Spearman（控制 stage 14 同区室共享轴供体分数，秩残差）；
    随机基因集零分布（同大小随机基因、按模型自身符号定向 —— 保留 DKD head 的整体方向，检验「这些基因」是否特殊）。"""
    if EGFR not in pb.donors:
        return pd.DataFrame(), pd.DataFrame()
    don = pb.rows.groupby("donor").indices
    Xd = np.vstack([np.asarray(pb.X[idx].sum(0)).ravel() for idx in don.values()])
    L = pd.DataFrame(log2cpm(Xd), index=list(don), columns=pb.genes)
    eg = pb.donors[EGFR].map(egfr_mid).reindex(L.index)
    sh = pd.read_csv(ROOT / "results/14_kpmp/tissue_shared_score.tsv", sep="\t")
    sh = sh[sh.dataset == pb.name]
    grp = {"DKD": pb.group_donors("DKD"), "CKD_all": pb.group_donors("DKD").union(pb.group_donors("OTHER_CKD"))}
    rs, rg = [], []
    for name, sg in sets.items():
        comp = name.split("_")[1]
        a_d = sh[sh.axis == comp].set_index("donor").shared_score
        g = [x for x in sg.index if x in L.columns]
        eff = G[G.compartment == comp].set_index("gene").e_full
        pool = np.array([x for x in eff.index if x in L.columns and L[x].std() > 0])
        for gname, dn in grp.items():
            d = [x for x in dn if x in L.index and not np.isnan(eg.get(x, np.nan))]
            dz = L.loc[d]

            def score(genes, signs):
                Z = dz[genes]
                Z = (Z - Z.mean()) / (Z.std() + 1e-9)
                return Z.to_numpy() @ signs / len(genes)
            sc = score(g, sg[g].to_numpy(float))
            r, p = stats.spearmanr(sc, eg[d])
            ok = a_d.reindex(d).notna().to_numpy()
            rp, pp = stats.pearsonr(_rank_resid(sc[ok], a_d.reindex(d).to_numpy()[ok]),
                                    _rank_resid(eg[d].to_numpy()[ok], a_d.reindex(d).to_numpy()[ok]))
            null = []
            for _ in range(IC["n_random_sets"]):
                gg = rng.choice(pool, len(g), replace=False)
                null.append(stats.spearmanr(score(list(gg), np.sign(eff[gg].to_numpy())), eg[d])[0])
            null = np.array(null)
            rs.append({"dataset": pb.name, "set": name, "donors": gname, "n_donors": len(d), "n_genes": len(g),
                       "spearman_score_egfr": r, "p": p, "partial_spearman_adj_shared": rp, "p_partial": pp,
                       "n_donors_partial": int(ok.sum()), "null_median": float(np.median(null)),
                       "pct_more_negative_than_random": float((null > r).mean()),
                       "spearman_shared_egfr": stats.spearmanr(a_d.reindex(d).to_numpy()[ok], eg[d].to_numpy()[ok])[0]})
            if name.startswith("key_"):
                pr = [stats.spearmanr(L.loc[d, x], eg[d]) for x in g]
                t = pd.DataFrame({"dataset": pb.name, "set": name, "donors": gname, "gene": g, "sign_model": sg[g].values,
                                  "spearman_egfr": [x[0] for x in pr], "p": [x[1] for x in pr]})
                t["fdr"] = bh(t.p.fillna(1).values)
                rg.append(t)
    return pd.DataFrame(rs), pd.concat(rg) if rg else pd.DataFrame()


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    sets, G = gene_sets()
    allg = sorted(set().union(*[set(s.index) for s in sets.values()]))
    DE = pd.read_csv(ROOT / "results/14_kpmp/de_kpmp.tsv.gz", sep="\t")
    DE = DE[(DE.level == "coarse") & DE.gene.isin(allg) & DE.contrast.isin(IC["kpmp_contrasts"])]
    res = {k: [] for k in ("localisation", "ref_expression", "gene_direction", "set_direction", "egfr_sets", "egfr_genes")}
    only_egfr = "--egfr-only" in sys.argv
    if only_egfr:                                  # 只重算 eGFR 表（其余表保持不变）
        res = {k: [] for k in ("egfr_sets", "egfr_genes")}
    for name in IC["kpmp_datasets"]:
        pb = Pseudobulk(name, K["datasets"][name], K)
        if not only_egfr:
            rows, X = pb.aggregate(K["map_to_stage13"])
            loc, long = localisation(pb, rows, X, allg)
            res["localisation"].append(loc), res["ref_expression"].append(long)
            res["gene_direction"].append(direction_rows(DE[DE.dataset == name], loc, sets, G))
            res["set_direction"].append(set_direction(pb, rows, X, sets, G, rng))
        a, b = egfr_tests(pb, sets, G, np.random.default_rng(CFG["seed"] + 18))
        res["egfr_sets"].append(a), res["egfr_genes"].append(b)
        print(name, "done", flush=True)
    outs = []
    for k, v in res.items():
        t = pd.concat(v, ignore_index=True)
        f = OUT / f"{k}.tsv"
        t.to_csv(f, sep="\t", index=False)
        outs.append(f)
        res[k] = t
    outs = [OUT / f"{k}.tsv" for k in ("localisation", "ref_expression", "gene_direction", "set_direction",
                                       "egfr_sets", "egfr_genes")]
    write_provenance("18_interpret/kpmp", [W / "gene_effects.tsv.gz", ROOT / "results/14_kpmp/de_kpmp.tsv.gz"]
                     + [ROOT / K["pseudobulk_cache"] / n / "source.json" for n in IC["kpmp_datasets"]], outs,
                     CFG["seed"], {"n_perm": IC["n_perm"], "celltype_mapping": "kpmp.map_to_stage13 (coarse)"})
    with pd.option_context("display.width", 250):
        if not only_egfr:
            sd = res["set_direction"]
            print(sd[sd.set.str.startswith("key")].round(3).to_string())
        print(res["egfr_sets"].round(3).to_string())


if __name__ == "__main__":
    main()
