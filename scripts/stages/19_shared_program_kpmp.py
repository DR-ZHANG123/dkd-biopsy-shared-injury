"""Stage 19c：KPMP 供体水平 —— SCP 基因在各细胞类型内 CKD vs 健康参考的方向与效应、组成 vs 细胞内状态分解、
取材（手术 vs 经皮活检）敏感性、供体 eGFR。供体为重复单位，全部 P 为供体标签置换。

对比：CKD_vs_REF（主）；CKD_perc_vs_REF_perc（取材匹配：两组都是经皮穿刺活检，仅 snRNA 有健康经皮活检）；
      DKD_vs_REF、OTHER_CKD_vs_REF（分病种）；REF_surg_vs_REF_perc（健康供体内的取材效应，不涉及疾病）。
基因分类（每个核心基因，按其参考最高表达类型 home）：
  state        home 类型内同向改变（|log2FC| > lfc_min 且置换 P < state_max_p）
  composition  home 内无同向改变，但基因类型特异（top_share ≥ comp_min_top_share）且 home 丰度同向改变（P < 0.05）
  state+composition / opposite_state / unresolved
产出 results/19_shared_program/kpmp/
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
from lib.kpmp import Pseudobulk  # noqa: E402
from lib.kpmp_stats import abundance_test, direction_test, hedges_g, label_perms, log2cpm  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from lib.scp_core import OUT, SP, load_core_table  # noqa: E402
from lib.scp_kpmp import Tissue, donor_groups, type_de  # noqa: E402
from lib.stats import auc_rows  # noqa: E402

CFG = load_config()
K, SK = CFG["kpmp"], SP["kpmp"]
O = OUT / "kpmp"
CONTRASTS = [("CKD", "REF"), ("CKD_perc", "REF_perc"), ("DKD", "REF"), ("OTHER_CKD", "REF"), ("REF_surg", "REF_perc")]
EGFR = "Baseline eGFR (ml/min/1.73m2) (Binned)"


def celltype_tests(pb, rows, X, grp, core, metas, rng):
    de, sets = [], []
    cg = pb.genes.get_indexer(sorted(set(core.gene)))
    cg_names = pb.genes[cg[cg >= 0]]
    for a, b in CONTRASTS:
        da, db = grp[a], grp[b]
        if len(da) < K["min_donors"] or len(db) < K["min_donors"]:
            continue
        ok = rows.donor.isin(da.union(db)) & (rows.n_cells >= K["min_cells"])
        r = rows[ok].reset_index(drop=True)
        L = log2cpm(X[np.flatnonzero(ok.values)])
        for ct, g in r.groupby("cell_type"):
            ia = g.index[g.donor.isin(da)].to_numpy()
            ib = g.index[g.donor.isin(db)].to_numpy()
            if len(ia) < K["min_donors"] or len(ib) < K["min_donors"]:
                continue
            Lc = L[:, cg[cg >= 0]]
            res = type_de(Lc, ia, ib, SK["n_perm"], rng)
            mexp = Lc[np.r_[ia, ib]].mean(0)
            de.append(pd.DataFrame({"dataset": pb.name, "contrast": f"{a}_vs_{b}", "cell_type": ct, "gene": cg_names,
                                    "n_A": len(ia), "n_B": len(ib), "mean_log2cpm": mexp, **res}))
            keep = L[np.r_[ia, ib]].mean(0) >= SK["min_expr_log2cpm"]
            Lab = np.vstack([L[ia][:, keep], L[ib][:, keep]])
            for comp, M in metas.items():
                bg = M.set_index("gene").g_re
                top = core[core.compartment == comp].set_index("gene").g_re
                t = direction_test(Lab, pb.genes[keep], len(ia), top, bg, K["lfc_min"], SK["n_perm"], rng)
                if t:
                    sets.append({"dataset": pb.name, "compartment": comp, "contrast": f"{a}_vs_{b}", "cell_type": ct,
                                 "n_A": len(ia), "n_B": len(ib)} | t)
    return pd.concat(de), pd.DataFrame(sets)


def abundance(pb, T: Tissue, grp, rng) -> pd.DataFrame:
    out = []
    lg = np.log(np.clip(T.P, 1e-4, 1 - 1e-4) / (1 - np.clip(T.P, 1e-4, 1 - 1e-4)))
    d = pd.Index(T.donors)
    for a, b in CONTRASTS:
        ia, ib = d.get_indexer(grp[a].intersection(d)), d.get_indexer(grp[b].intersection(d))
        if len(ia) < K["min_donors"] or len(ib) < K["min_donors"]:
            continue
        for k, t in enumerate(T.types):
            out.append({"dataset": pb.name, "contrast": f"{a}_vs_{b}", "cell_type": t, "n_A": len(ia), "n_B": len(ib),
                        "mean_frac_A": T.P[ia, k].mean(), "mean_frac_B": T.P[ib, k].mean(),
                        **abundance_test(lg[ia, k], lg[ib, k], 500, SK["n_perm"], rng)})
    return pd.DataFrame(out)


def score_contrast(sc: np.ndarray, donors: pd.Index, da, db, rng) -> dict:
    ia, ib = donors.get_indexer(da.intersection(donors)), donors.get_indexer(db.intersection(donors))
    x = np.r_[sc[ia], sc[ib]]
    assign = label_perms(len(x), len(ia), SK["n_perm"], rng)
    diff = np.array([x[r].mean() - x[~r].mean() for r in assign])
    y = np.r_[np.ones(len(ia)), np.zeros(len(ib))].astype(int)
    return {"n_A": len(ia), "n_B": len(ib), "mean_A": x[:len(ia)].mean(), "mean_B": x[len(ia):].mean(),
            "delta": diff[0], "hedges_g": float(hedges_g(x[:len(ia), None], x[len(ia):, None])[0]),
            "auroc": float(auc_rows(y, x[None])[0]),
            "p_perm": float((1 + (np.abs(diff[1:]) >= abs(diff[0]) - 1e-12).sum()) / len(diff))}


def decomposition(pb, T: Tissue, comp, up, dn, grp, rng):
    prof = T.profiles()
    donors = pd.Index(T.donors)
    long, tests = [], []
    for scen, E in prof.items():
        sc = T.score(E, up, dn)
        long.append(pd.DataFrame({"dataset": pb.name, "compartment": comp, "scenario": scen, "donor": T.donors,
                                  "score": sc}))
        for a, b in CONTRASTS[:4]:
            if len(grp[a]) >= K["min_donors"] and len(grp[b]) >= K["min_donors"]:
                tests.append({"dataset": pb.name, "compartment": comp, "scenario": scen, "contrast": f"{a}_vs_{b}",
                              **score_contrast(sc, donors, grp[a], grp[b], rng)})
    return pd.concat(long), pd.DataFrame(tests)


def egfr_tests(pb, T: Tissue, comp, up, dn, grp, rng) -> list[dict]:
    if EGFR not in pb.donors:
        return []
    mid = import_module("18_interpret_kpmp").egfr_mid
    eg = pb.donors[EGFR].map(mid).reindex(T.donors).to_numpy(float)
    prof = T.profiles()
    out = []
    for gname, dn_ in (("CKD", grp["CKD"]), ("DKD", grp["DKD"]), ("all_with_egfr", pd.Index(T.donors))):
        m = np.isin(T.donors, dn_) & np.isfinite(eg)
        for scen in ("full", "comp_only", "state_only"):
            sc = T.score(prof[scen], up, dn)
            r, p = stats.spearmanr(sc[m], eg[m])
            row = {"dataset": pb.name, "compartment": comp, "donors": gname, "scenario": scen, "n": int(m.sum()),
                   "spearman": r, "p": p}
            if scen == "full":
                R = stats.rankdata(prof[scen][m], axis=1) / prof[scen].shape[1]
                null = []
                for _ in range(SP["replicate"]["n_random_sets"]):
                    pick = rng.choice(R.shape[1], len(up) + len(dn), replace=False)
                    null.append(stats.spearmanr(R[:, pick[:len(up)]].mean(1) - R[:, pick[len(up):]].mean(1), eg[m])[0])
                row |= {"null_median": float(np.median(null)), "pct_more_negative_than_random": float((np.array(null) > r).mean())}
            out.append(row)
    return out


def procurement(pb, T: Tissue, grp, rng) -> pd.DataFrame:
    Lt = T.tissue_log2cpm()
    out = []
    for a, b in (("REF_surg", "REF_perc"), ("CKD_perc", "REF_perc"), ("CKD", "REF")):
        da, db = grp[a].intersection(Lt.index), grp[b].intersection(Lt.index)
        if len(da) < K["min_donors"] or len(db) < K["min_donors"]:
            continue
        La, Lb = Lt.loc[da].to_numpy(), Lt.loc[db].to_numpy()
        res = type_de(np.vstack([La, Lb]), np.arange(len(da)), np.arange(len(da), len(da) + len(db)), SK["n_perm"], rng)
        out.append(pd.DataFrame({"dataset": pb.name, "contrast": f"{a}_vs_{b}", "gene": Lt.columns, "n_A": len(da),
                                 "n_B": len(db), **res}))
    return pd.concat(out) if out else pd.DataFrame()


def classify(core, loc, de, ab) -> pd.DataFrame:
    rows = []
    for (ds, con), d in de.groupby(["dataset", "contrast"]):
        if con not in ("CKD_vs_REF", "CKD_perc_vs_REF_perc"):
            continue
        d = d.set_index(["gene", "cell_type"])
        a = ab[(ab.dataset == ds) & (ab.contrast == con)].set_index("cell_type")
        lo = loc[loc.dataset == ds].set_index("gene")
        for _, c in core.iterrows():
            if c.gene not in lo.index:
                continue
            l = lo.loc[c.gene]
            home, sgn = l.top_type, np.sign(c.g_re)
            r = d.loc[(c.gene, home)] if (c.gene, home) in d.index else None
            st = r is not None and np.sign(r.log2fc) == sgn and abs(r.log2fc) > SK["lfc_min"] and r.p_perm < SK["state_max_p"]
            opp = r is not None and np.sign(r.log2fc) == -sgn and abs(r.log2fc) > SK["lfc_min"] and r.p_perm < SK["state_max_p"]
            dg = d.xs(c.gene, level="gene") if c.gene in d.index.get_level_values(0) else pd.DataFrame()
            dg = dg[dg.mean_log2cpm >= SK["min_expr_log2cpm"]] if len(dg) else dg
            n_state_any = int(((np.sign(dg.log2fc) == sgn) & (dg.log2fc.abs() > SK["lfc_min"]) & (dg.p_perm < SK["state_max_p"])).sum()) if len(dg) else 0
            ah = a.loc[home] if home in a.index else None
            comp_ok = (l.top_share >= SK["comp_min_top_share"] and ah is not None
                       and np.sign(ah.hedges_g) == sgn and ah.perm_p < 0.05)
            cls = ("state+composition" if st and comp_ok else "state" if st else "composition" if comp_ok
                   else "opposite_state" if opp else "unresolved")
            rows.append({"dataset": ds, "contrast": con, "compartment": c.compartment, "gene": c.gene,
                         "direction": c.direction, "g_re": c.g_re, "home_type": home, "top_share": l.top_share,
                         "tau": l.tau, "home_log2fc": getattr(r, "log2fc", np.nan), "home_p_perm": getattr(r, "p_perm", np.nan),
                         "home_abundance_g": getattr(ah, "hedges_g", np.nan), "home_abundance_p": getattr(ah, "perm_p", np.nan),
                         "n_types_state_concordant": n_state_any, "class": cls})
    return pd.DataFrame(rows)


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    core = load_core_table()
    metas = {c: pd.read_csv(OUT / "core" / f"meta_{c}.tsv.gz", sep="\t") for c in SP["sources"]}
    loc = pd.read_csv(OUT / "cells" / "kpmp_localisation_all_genes.tsv.gz", sep="\t")
    res = {k: [] for k in ("gene_type_de", "set_direction", "abundance", "decomp_scores", "decomp_tests", "egfr",
                           "procurement", "groups")}
    for name in SK["datasets"]:
        pb = Pseudobulk(name, K["datasets"][name], K)
        rows, X = pb.aggregate(K["map_to_stage13"])
        grp = donor_groups(pb, SK)
        res["groups"].append(pd.DataFrame([{"dataset": name, "group": g, "n_donors": len(v)} for g, v in grp.items()]))
        de, sd = celltype_tests(pb, rows, X, grp, core, metas, rng)
        res["gene_type_de"].append(de), res["set_direction"].append(sd)
        uni_all = sorted(set().union(*[set(m.gene) for m in metas.values()]))
        T_all = Tissue(rows, X, pb.genes, uni_all, grp["REF"], K["min_cells"])
        res["abundance"].append(abundance(pb, T_all, grp, rng))
        res["procurement"].append(procurement(pb, T_all, grp, rng))
        del T_all
        for comp, M in metas.items():
            T = Tissue(rows, X, pb.genes, M.gene.tolist(), grp["REF"], K["min_cells"])
            c = core[core.compartment == comp]
            up, dn = c.gene[c.direction == "up"].tolist(), c.gene[c.direction == "down"].tolist()
            a, b = decomposition(pb, T, comp, up, dn, grp, rng)
            res["decomp_scores"].append(a), res["decomp_tests"].append(b)
            res["egfr"].append(pd.DataFrame(egfr_tests(pb, T, comp, up, dn, grp, rng)))
            del T
        print(name, "done", flush=True)
    R = {k: pd.concat(v, ignore_index=True) for k, v in res.items()}
    R["gene_class"] = classify(core, loc, R["gene_type_de"], R["abundance"])
    pr = R["procurement"]
    ann = core[["compartment", "gene", "direction", "g_re"]].copy()
    for con, tag in (("REF_surg_vs_REF_perc", "proc"), ("CKD_perc_vs_REF_perc", "matched"), ("CKD_vs_REF", "ckd")):
        p = pr[(pr.dataset == "snRNA") & (pr.contrast == con)].set_index("gene")
        ann[f"{tag}_g"] = ann.gene.map(p.g)
        ann[f"{tag}_p_perm"] = ann.gene.map(p.p_perm)
    ann["procurement_sensitive"] = (np.sign(ann.proc_g) == -np.sign(ann.g_re)) & (ann.proc_g.abs() >= SK["proc_min_abs_g"])
    ann["matched_concordant"] = np.sign(ann.matched_g) == np.sign(ann.g_re)
    R["core_procurement"] = ann
    outs = []
    for k, t in R.items():
        f = O / (f"{k}.tsv.gz" if k in ("gene_type_de", "procurement", "decomp_scores") else f"{k}.tsv")
        t.to_csv(f, sep="\t", index=False)
        outs.append(f)
    write_provenance("19_shared_program/kpmp", [OUT / "core/core_genes.tsv"] +
                     [ROOT / K["pseudobulk_cache"] / n / "source.json" for n in SK["datasets"]], outs, CFG["seed"],
                     {"contrasts": [f"{a}_vs_{b}" for a, b in CONTRASTS], "n_perm": SK["n_perm"]})
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(R["groups"].to_string())
        dt = R["decomp_tests"]
        print(dt[dt.scenario.isin(["full", "comp_only", "state_only"])].round(3).to_string())
        print(R["gene_class"].groupby(["dataset", "contrast", "compartment", "direction", "class"]).size().unstack().to_string())
        print(R["egfr"].round(3).to_string())
        print(ann.groupby(["compartment", "direction"])[["procurement_sensitive", "matched_concordant"]].mean().round(3))


if __name__ == "__main__":
    main()
