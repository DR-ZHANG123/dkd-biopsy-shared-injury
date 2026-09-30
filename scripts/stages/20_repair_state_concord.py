"""Stage 20b：SCP 的小管细胞内状态成分 vs KPMP 修复失败状态程序（一致性与供体水平回归）。

(1) 谱系内疾病变化：PT（全部状态）/ 正常 PT、TAL / 正常 TAL 的供体 pseudobulk，CKD vs REF（及取材匹配 CKD_perc vs REF_perc）
    逐基因 log2FC；与状态程序（stage 20a，供体内配对，未用疾病标签）的逐基因 Spearman（全部表达基因；SCP 核心基因）；
    零分布 = 供体标签置换后重算的 log2FC（n_perm）。
(2) bulk SCP（TUB meta g_re，与 KPMP 完全独立）与状态程序：逐基因 Spearman；SCP 上 / 下调核心 ∩ 程序上 / 下调的 Fisher OR；
    SCP 核心集上程序 log2FC 均值 vs 表达量分层匹配随机集。
(3) 供体回归：PT（或 TAL）谱系 pseudobulk 的 SCP 分数 ~ logit(修复失败状态占比) / 正常细胞 SCP 分数 / 两者；R² 与置换 P。
产出 results/20_repair_state/concord/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.kpmp import Pseudobulk  # noqa: E402
from lib.kpmp_stats import label_perms, log2cpm  # noqa: E402
from lib.repair_state import (CFG, K, OUT, RS, category_index, donor_categories, expr_matched_null,  # noqa: E402
                              logit, rank_score, repair_label, state_pseudobulk)
from lib.repro import ROOT, set_global_seed, write_provenance  # noqa: E402
from lib.scp_core import OUT as SCP_OUT, load_core  # noqa: E402

O = OUT / "concord"
PROGRAM_STATES = {"PT": ["aPT", "frPT", "rfPT", "dPT", "cycPT"], "TAL": ["aTAL", "frTAL", "rfTAL", "dTAL", "cycTAL"]}


def contrasts(pb, cat) -> dict[str, tuple[pd.Index, pd.Index]]:
    ref, ckd = category_index(cat, "REF"), category_index(cat, "CKD")
    out = {"CKD_vs_REF": (ckd, ref)}
    pc = CFG["shared_program"]["kpmp"]["procurement_col"][pb.name]
    perc = pb.donors.index[pb.donors[pc].astype(str) == CFG["shared_program"]["kpmp"]["percutaneous"]]
    if len(ref.intersection(perc)) >= RS["min_donors"]:
        out["CKD_perc_vs_REF_perc"] = (ckd.intersection(perc), ref.intersection(perc))
    return out


def rank_rows(M: np.ndarray) -> np.ndarray:
    R = M.argsort(1).argsort(1).astype(float)
    R -= R.mean(1, keepdims=True)
    return R / (np.sqrt((R ** 2).sum(1, keepdims=True)) + 1e-12)


def lineage_de(pbs, lab, genes, da, db, rng):
    d, X, _ = pbs[lab]
    ia, ib = d.get_indexer(da.intersection(d)), d.get_indexer(db.intersection(d))
    if min(len(ia), len(ib)) < RS["min_donors"]:
        return None
    L = log2cpm(X[np.r_[ia, ib]])
    keep = L.mean(0) >= RS["program"]["min_log2cpm"]
    L = L[:, keep]
    assign = label_perms(len(L), len(ia), RS["n_perm"], rng)
    A = assign / assign.sum(1, keepdims=True)
    B = ~assign / (~assign).sum(1, keepdims=True)
    D = (A - B) @ L                                                  # (1 + n_perm) × 基因；第 0 行 = 观测
    p = (1 + (np.abs(D[1:]) >= np.abs(D[0]) - 1e-12).sum(0)) / len(D)
    return pd.Series(D[0], index=genes[keep]), D, pd.Series(p, index=genes[keep]), pd.Series(L.mean(0), index=genes[keep])


def concord_rows(ds, lin, lab, con, D, gk, prog, core_up, core_dn) -> list[dict]:
    rows = []
    for st in PROGRAM_STATES[lin]:
        pr = prog[(prog.dataset == ds) & (prog.lineage == lin) & (prog.state == st)].set_index("gene").log2fc
        if pr.empty:
            continue
        for gs_name, gset in (("all_expressed", None), ("scp_core", core_up + core_dn)):
            g = gk.intersection(pr.index) if gset is None else gk.intersection(pr.index).intersection(gset)
            if len(g) < 20:
                continue
            ci = gk.get_indexer(g)
            Rd = rank_rows(D[:, ci])
            rp = stats.rankdata(pr[g].to_numpy())
            rp = (rp - rp.mean()) / np.sqrt(((rp - rp.mean()) ** 2).sum())
            rho = Rd @ rp
            rows.append({"dataset": ds, "lineage": lin, "lineage_cells": lab.split("|")[1], "contrast": con,
                         "program": st, "gene_set": gs_name, "n_genes": len(g), "spearman": rho[0],
                         "null_mean": rho[1:].mean(), "null_sd": rho[1:].std(),
                         "p_perm": (1 + (rho[1:] >= rho[0]).sum()) / len(rho)})
    return rows


def bulk_overlap(prog, meta, core_up, core_dn, expr, rng) -> list[dict]:
    rows = []
    g_re = meta.set_index("gene").g_re
    for (ds, lin, st), t in prog.groupby(["dataset", "lineage", "state"]):
        t = t.set_index("gene")
        ex = expr.get((ds, lin))
        bg = t.index.intersection(g_re.index)
        r, p = stats.spearmanr(t.loc[bg, "log2fc"], g_re[bg])
        base = {"dataset": ds, "lineage": lin, "program": st, "n_background": len(bg),
                "spearman_program_vs_bulk_scp_g": r, "p_spearman_nominal": p}
        for dr, cs in (("up", core_up), ("down", core_dn)):
            cs = [x for x in cs if x in bg]
            call = t.loc[bg, "call"].fillna("")
            a = int(np.isin(cs, call.index[call == dr]).sum())
            b = len(cs) - a
            c = int((call == dr).sum()) - a
            d = len(bg) - a - b - c
            orr, pf = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            nul = expr_matched_null(t.log2fc, ex, cs, RS["n_random_sets"], RS["n_expr_bins"], rng) if ex is not None else {}
            rows.append(base | {"scp_direction": dr, "n_scp_core_expr": len(cs), "n_overlap_same_dir": a,
                                "frac_core_in_program": a / max(len(cs), 1), "frac_background_in_program": (a + c) / len(bg),
                                "fisher_or": orr, "fisher_p_greater": pf,
                                "frac_core_lfc_same_sign": float(np.mean(np.sign(t.loc[cs, "log2fc"]) == (1 if dr == "up" else -1))),
                                **{f"matched_{k}": v for k, v in nul.items()}})
    return rows


def regression(pb, pbs, cat, fr, up, dn, rng) -> tuple[list[dict], pd.DataFrame]:
    rows, scores = [], []
    for lin in RS["lineages"]:
        dA, XA, _ = pbs[f"{lin}|all"]
        dN, XN, _ = pbs[f"{lin}|normal"]
        LA = log2cpm(XA)
        ex = LA[dA.isin(category_index(cat, "REF"))].mean(0) >= RS["program"]["min_log2cpm"]
        genes = pb.genes[ex]
        sA = pd.Series(rank_score(LA[:, ex], genes, up, dn), index=dA)
        sN = pd.Series(rank_score(log2cpm(XN)[:, ex], genes, up, dn), index=dN)
        f = fr[(fr.lineage == lin)].set_index(["state", "donor"]).frac
        rf = f.xs(repair_label(lin)).reindex(dA)
        alt = f.xs("altered").reindex(dA)
        d = pd.DataFrame({"dataset": pb.name, "lineage": lin, "donor": dA, "category": cat.reindex(dA).to_numpy(),
                          "scp_lineage": sA.to_numpy(), "scp_normal_cells": sN.reindex(dA).to_numpy(),
                          "repair_frac": rf.to_numpy(), "altered_frac": alt.to_numpy()})
        scores.append(d)
        for subset, mask in (("REF+CKD", d.category.isin(["REF"] + RS["ckd"])), ("CKD", d.category.isin(RS["ckd"])),
                             ("all_categorised", d.category.notna())):
            t = d[mask & d.scp_normal_cells.notna() & d.repair_frac.notna()]
            if len(t) < 10:
                continue
            y = t.scp_lineage.to_numpy()
            preds = {"logit_repair_frac": logit(t.repair_frac, 1e-3)[:, None],
                     "logit_altered_frac": logit(t.altered_frac, 1e-3)[:, None],
                     "normal_cell_scp": t.scp_normal_cells.to_numpy()[:, None]}
            preds["repair_frac+normal_cell_scp"] = np.c_[preds["logit_repair_frac"], preds["normal_cell_scp"]]
            for pn, Xp in preds.items():
                r2, pnull = _r2_perm(Xp, y, rng)
                rows.append({"dataset": pb.name, "lineage": lin, "donors": subset, "predictor": pn, "n": len(t),
                             "r2": r2, "p_perm": pnull,
                             "spearman": stats.spearmanr(Xp[:, 0], y)[0] if Xp.shape[1] == 1 else np.nan})
    return rows, pd.concat(scores)


def _r2_perm(X, y, rng) -> tuple[float, float]:
    Z = np.c_[np.ones(len(y)), X]

    def r2(yy):
        b = np.linalg.lstsq(Z, yy, rcond=None)[0]
        return 1 - ((yy - Z @ b) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    obs = r2(y)
    null = np.array([r2(rng.permutation(y)) for _ in range(RS["n_perm"])])
    return float(obs), float((1 + (null >= obs).sum()) / (1 + len(null)))


def main() -> None:
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    O.mkdir(parents=True, exist_ok=True)
    comp = RS["compartment"]
    up, dn = load_core(comp)
    meta = pd.read_csv(SCP_OUT / "core" / f"meta_{comp}.tsv.gz", sep="\t")
    prog = pd.read_csv(OUT / "programs" / "programs.tsv.gz", sep="\t")
    prog = prog[prog.subset == "all"]
    fr = pd.read_csv(OUT / "programs" / "fractions.tsv.gz", sep="\t", dtype={"donor": str})
    conc, lde, regs, dsc, expr = [], [], [], [], {}
    for name in RS["datasets"]:
        pb = Pseudobulk(name, K["datasets"][name], K)
        cat = donor_categories(pb, name)
        pbs = state_pseudobulk(pb, RS["min_cells"])
        for lin in RS["lineages"]:
            for cells in ("all", "normal"):
                lab = f"{lin}|{cells}"
                for con, (da, db) in contrasts(pb, cat).items():
                    res = lineage_de(pbs, lab, pb.genes, da, db, rng)
                    if res is None:
                        continue
                    lfc, D, p, mexp = res
                    if cells == "all" and con == "CKD_vs_REF":
                        expr[(name, lin)] = mexp
                    lde.append(pd.DataFrame({"dataset": name, "lineage": lin, "lineage_cells": cells, "contrast": con,
                                             "gene": lfc.index, "log2fc": lfc.to_numpy(), "p_perm": p.to_numpy(),
                                             "mean_log2cpm": mexp.to_numpy(), "scp_core": np.where(
                                                 lfc.index.isin(up), "up", np.where(lfc.index.isin(dn), "down", ""))}))
                    conc += concord_rows(name, lin, lab, con, D, lfc.index, prog, up, dn)
        r, s = regression(pb, pbs, cat, fr[fr.dataset == name], up, dn, rng)
        regs += r
        dsc.append(s)
        print(name, "done", flush=True)
    ov = pd.DataFrame(bulk_overlap(prog, meta, up, dn, expr, rng))
    outs = {"concordance.tsv": pd.DataFrame(conc), "lineage_de.tsv.gz": pd.concat(lde), "bulk_overlap.tsv": ov,
            "regression.tsv": pd.DataFrame(regs), "donor_lineage_scores.tsv": pd.concat(dsc)}
    for f, t in outs.items():
        t.to_csv(O / f, sep="\t", index=False)
    write_provenance("20_repair_state/concord", [SCP_OUT / "core/core_genes.tsv", SCP_OUT / f"core/meta_{comp}.tsv.gz",
                                                 OUT / "programs/programs.tsv.gz", OUT / "programs/fractions.tsv.gz"],
                     [O / f for f in outs], CFG["seed"], {"n_perm": RS["n_perm"], "compartment": comp})
    with pd.option_context("display.width", 250, "display.max_rows", 400, "display.max_columns", 30):
        print(outs["concordance.tsv"].round(3).to_string())
        print(ov[["dataset", "lineage", "program", "scp_direction", "spearman_program_vs_bulk_scp_g", "n_scp_core_expr",
                  "n_overlap_same_dir", "fisher_or", "fisher_p_greater", "frac_core_lfc_same_sign", "matched_z",
                  "matched_p_upper", "matched_p_lower"]].round(3).to_string())
        print(outs["regression.tsv"].round(3).to_string())


if __name__ == "__main__":
    main()
