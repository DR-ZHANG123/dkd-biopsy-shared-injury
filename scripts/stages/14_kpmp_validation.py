"""Stage 14：KPMP 单细胞 / 单核参照与 stage 13 细胞类型结论的独立验证。

a) 仅用 KPMP 健康参考供体构建细胞类型标志（供体 × 细胞类型 pseudobulk，特异性 top-n，跨供体二分重抽样稳定），
   用 stage 13 的算法在 stage 13 的逐基因效应表上重算富集，并与旧参照（GSE131882 / GSE209781 对照供体）对照。
b) 独立验证（供体为重复单位）：KPMP DKD vs 其他 CKD / 健康参考 / AKI 的逐细胞类型 pseudobulk 差异；
   bulk DKD 特异基因（DKD_vs_PAT_adj）方向在各细胞类型中的复现（供体标签置换 P）；供体水平细胞比例。
输出 results/14_kpmp/。用法：python scripts/stages/14_kpmp_validation.py [--datasets snRNA,scRNA] [--skip-perm]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.kpmp import Pseudobulk  # noqa: E402
from lib.kpmp_report import BULK_FOR, claims_table, composition, old_vs_new, report  # noqa: E402
from lib.kpmp_stats import abundance_test, build_markers, direction_test, enrichment, log2cpm, welch_de  # noqa: E402
from lib.kpmp_tissue import marker_expression, old_marker_profile, tissue_replication  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
K = CFG["kpmp"]
OUT = ROOT / "results" / "14_kpmp"
S13 = ROOT / "results" / "13_celltype"


def load_stage13():
    E = pd.read_csv(S13 / "gene_effects_by_unit.tsv.gz", sep="\t")
    meta = pd.read_csv(S13 / "gene_effects_meta.tsv.gz", sep="\t")
    old = pd.read_csv(S13 / "celltype_enrichment.tsv", sep="\t")
    om = pd.read_csv(S13 / "celltype_markers.tsv", sep="\t")
    old_markers = {c: pd.Index(om.gene[om.cell_type == c]) for c in CFG["celltype"]["keep"] if (om.cell_type == c).any()}
    return E, meta, old, old_markers


def expressed(rows: pd.DataFrame, X, donors: pd.Index) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """细胞类型 → (合格行, log2CPM)；合格 = 供体属于 donors 且 n_cells ≥ min_cells。"""
    ok = rows.donor.isin(donors) & (rows.n_cells >= K["min_cells"])
    L = log2cpm(X[np.flatnonzero(ok.values)])
    r = rows[ok].reset_index(drop=True)
    return {c: (g, L[g.index.values]) for c, g in r.groupby("cell_type")}


def reference_markers(pb: Pseudobulk, rows, X, universe: pd.Index, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = pb.group_donors(K["marker_group"])
    thr = rows.cell_type.map(K.get("marker_min_cells", {})).fillna(K["min_cells"])
    ok = rows.donor.isin(ref) & (rows.n_cells >= thr)
    L = pd.DataFrame(log2cpm(X[np.flatnonzero(ok.values)]), columns=pb.genes)
    L = L.loc[:, L.columns.isin(universe)]
    return build_markers(L, rows[ok].reset_index(drop=True), K, rng)


def marker_dict(mk: pd.DataFrame, order) -> dict[str, pd.Index]:
    return {c: pd.Index(mk.gene[mk.cell_type == c]) for c in order if (mk.cell_type == c).any()}


def de_and_direction(pb, levels, meta, n_perm, rng) -> tuple[list[pd.DataFrame], list[dict]]:
    de, dirs = [], []
    for a, b in K["contrasts"]:
        da, db = pb.group_donors(a), pb.group_donors(b)
        if len(da.intersection(db)):
            raise ValueError(f"{a} 与 {b} 供体重叠")
        for level, (rows, X) in levels.items():
            per = expressed(rows, X, da.union(db))
            for ct, (r, L) in per.items():
                ia, ib = np.flatnonzero(r.donor.isin(da).values), np.flatnonzero(r.donor.isin(db).values)
                if len(ia) < K["min_donors"] or len(ib) < K["min_donors"]:
                    continue
                keep = L[np.r_[ia, ib]].mean(0) >= K["min_expr_log2cpm"]
                genes = pb.genes[keep]
                LA, LB = L[ia][:, keep], L[ib][:, keep]
                res = welch_de(LA, LB)
                de.append(pd.DataFrame({"dataset": pb.name, "contrast": f"{a}_vs_{b}", "level": level, "cell_type": ct,
                                        "gene": genes, "n_A": len(ia), "n_B": len(ib)} | res))
                Lab = np.vstack([LA, LB])
                for bc in BULK_FOR.get(a, []):
                    for comp in ("GLOM", "TUB"):
                        m = meta[(meta.compartment == comp) & (meta.contrast == bc)].set_index("gene")
                        if m.empty:
                            continue
                        top = m[m.sign_consistency == 1].g_mean.abs().nlargest(K["n_top_validate"]).index
                        t = direction_test(Lab, genes, len(ia), m.g_mean[top], m.g_mean, K["lfc_min"], n_perm, rng)
                        if t:
                            dirs.append({"dataset": pb.name, "bulk_contrast": bc, "compartment": comp,
                                         "kpmp_contrast": f"{a}_vs_{b}", "level": level, "cell_type": ct,
                                         "n_A": len(ia), "n_B": len(ib)} | t)
    return de, dirs


def abundance(pb: Pseudobulk, rows, n_boot, n_perm, rng) -> tuple[pd.DataFrame, list[dict]]:
    tot = pb.donors.n_cells_total.astype(float)
    wide = rows.pivot_table(index="donor", columns="cell_type", values="n_cells", fill_value=0).reindex(tot.index, fill_value=0)
    fc = pb.fine_counts.reindex(tot.index, fill_value=0)
    glom = pd.DataFrame({k: fc.reindex(columns=v, fill_value=0).sum(1) for k, v in K["glom_fine"].items()})
    gtot = glom.sum(1)
    long = [pd.DataFrame({"donor": w.index.repeat(w.shape[1]), "cell_type": np.tile(w.columns, len(w)),
                          "denominator": den, "n_cells": w.values.ravel(), "n_total": np.repeat(t.values, w.shape[1])})
            for w, t, den in ((wide, tot, "all"), (glom[gtot >= K["min_glom_cells"]], gtot[gtot >= K["min_glom_cells"]], "glom"))]
    ab = pd.concat(long, ignore_index=True)
    ab["frac"] = ab.n_cells / ab.n_total
    ab.insert(0, "dataset", pb.name)
    ab["groups"] = ab.donor.map(lambda d: ",".join(g for g in pb.groups if pb.donors.at[d, g] in (True, "True")))
    tests = []
    for a, b in K["contrasts"]:
        da, db = pb.group_donors(a), pb.group_donors(b)
        for (ct, den), g in ab.groupby(["cell_type", "denominator"]):
            x = g.set_index("donor")
            lg = np.log((x.n_cells + 0.5) / (x.n_total - x.n_cells + 0.5))
            va, vb = lg.reindex(da).dropna().values, lg.reindex(db).dropna().values
            if len(va) < K["min_donors"] or len(vb) < K["min_donors"]:
                continue
            fa, fb = x.frac.reindex(da).dropna(), x.frac.reindex(db).dropna()
            tests.append({"dataset": pb.name, "contrast": f"{a}_vs_{b}", "cell_type": ct, "denominator": den,
                          "n_A": len(va), "n_B": len(vb), "median_A": fa.median(), "median_B": fb.median(),
                          "iqr_A": fa.quantile(.75) - fa.quantile(.25), "iqr_B": fb.quantile(.75) - fb.quantile(.25)}
                         | abundance_test(va, vb, n_boot, n_perm, rng))
    return ab, tests


def run_dataset(name: str, E, meta, old_markers, universe, n_perm: int, n_boot: int) -> dict:
    dcfg = K["datasets"][name]
    rng = np.random.default_rng(CFG["seed"])
    pb = Pseudobulk(name, dcfg, K)
    rows_c, Xc = pb.aggregate(K["map_to_stage13"])
    mk, z = reference_markers(pb, rows_c, Xc, universe, rng)
    markers = marker_dict(mk, K["map_to_stage13"])
    split = {k: v for k, v in K["map_to_stage13"].items() if k != "ENDO"} | K["endo_split"]
    rows_s, Xs = pb.aggregate(split)
    mk_s, _ = reference_markers(pb, rows_s, Xs, universe, rng)
    enr = [enrichment(E, meta, markers, n_perm, CFG["seed"]).assign(reference=name),
           enrichment(E, meta, marker_dict(mk_s, split), n_perm, CFG["seed"]).assign(reference=f"{name}_endosplit")]
    levels = {"coarse": (rows_c, Xc), "fine": pb.fine_level(K["fine_validate"])}
    de, dirs = de_and_direction(pb, levels, meta, n_perm, rng)
    ab, ab_tests = abundance(pb, rows_c, n_boot, n_perm, rng)
    per = {(a, b): expressed(rows_c, Xc, pb.group_donors(a).union(pb.group_donors(b))) for a, b in K["contrasts"]}
    mexp = marker_expression(pb, per, markers | marker_dict(mk_s, K["endo_split"]), K, n_perm, rng)
    t_eff, t_enr, t_rep, t_sc = tissue_replication(pb, meta, {f"kpmp_{name}": markers, "old": old_markers},
                                                   K, n_perm, CFG["seed"], rng)
    return {"pb": pb, "markers_kpmp.tsv": pd.concat([mk.assign(reference=name), mk_s[mk_s.cell_type.isin(K["endo_split"])]
                                                     .assign(reference=f"{name}_endosplit")]),
            "celltype_enrichment_kpmp.tsv": pd.concat(enr), "de_kpmp.tsv.gz": pd.concat(de),
            "direction_validation.tsv": pd.DataFrame(dirs), "abundance_by_donor.tsv": ab,
            "abundance_tests.tsv": pd.DataFrame(ab_tests), "marker_expression.tsv": pd.DataFrame(mexp),
            "tissue_effects.tsv.gz": t_eff, "tissue_enrichment.tsv": t_enr, "tissue_replication.tsv": t_rep,
            "tissue_shared_score.tsv": t_sc, "old_marker_profile.tsv": old_marker_profile(z, old_markers, name),
            "donor_composition.tsv": composition(pb, dcfg),
            "unmapped_fine_types.tsv": pb.unmapped(K["map_to_stage13"]).assign(dataset=name)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(K["datasets"]))
    ap.add_argument("--skip-perm", action="store_true")
    args = ap.parse_args()
    n_perm = 50 if args.skip_perm else K["n_perm"]
    n_boot = 100 if args.skip_perm else K["n_boot"]
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    E, meta, old, old_markers = load_stage13()
    universe = pd.Index(meta.gene.unique())
    # 算法核对：旧标志 + 本脚本富集函数应逐值复现 stage 13 的 celltype_enrichment.tsv
    chk = enrichment(E, meta, old_markers, CFG["celltype"]["n_perm"], CFG["seed"]).merge(
        old, on=["compartment", "contrast", "unit", "cell_type"], suffixes=("", "_old"))
    repro_err = float((chk.z - chk.z_old).abs().max())
    print(f"stage 13 富集复现：{len(chk)} 行，max|Δz| = {repro_err:.2e}")
    runs = {n: run_dataset(n, E, meta, old_markers, universe, n_perm, n_boot)
            for n in args.datasets.split(",") if (ROOT / K["datasets"][n]["file"]).is_file()}
    tables = {f: pd.concat([r[f] for r in runs.values()], ignore_index=True) for f in next(iter(runs.values())) if f != "pb"}
    mk = tables["markers_kpmp.tsv"]
    tables["marker_overlap.tsv"] = pd.DataFrame([
        {"reference": r, "cell_type": c, "n_new": len(set(g.gene)), "n_old": len(old_markers.get(c, [])),
         "n_overlap": len(set(g.gene) & set(old_markers.get(c, []))),
         "jaccard": len(set(g.gene) & set(old_markers.get(c, []))) / len(set(g.gene) | set(old_markers.get(c, [])))}
        for (r, c), g in mk.groupby(["reference", "cell_type"])])
    enr = tables["celltype_enrichment_kpmp.tsv"]
    tables["enrichment_old_vs_new.tsv"], tables["enrichment_concordance.tsv"] = old_vs_new(old, enr)
    tables["claims.tsv"] = claims_table(old, enr)
    outs = []
    for f, t in tables.items():
        t.to_csv(OUT / f, sep="\t", index=False)
        outs.append(OUT / f)
    report(tables, repro_err)
    if args.skip_perm:
        return
    extra = {"datasets": {n: {"file": K["datasets"][n]["file"], "dataset_id": K["datasets"][n]["dataset_id"],
                              "md5": r["pb"].source["md5"], "n_cells_kept": r["pb"].source["n_cells_kept"],
                              "cell_filter": r["pb"].source["cell_filter"],
                              "n_donors_by_group": {g: int(len(r["pb"].group_donors(g))) for g in r["pb"].groups}}
                          for n, r in runs.items()},
             "stage13_enrichment_reproduction_max_abs_dz": repro_err,
             "note_dissociation_bias": ("Cell proportions are relative capture, not absolute abundance: scRNA-seq of "
                                        "dissociated biopsies under-recovers podocytes and other glomerular cells and "
                                        "over-represents immune cells; snRNA-seq favours epithelial nuclei. Glomerular "
                                        "proportions use the four glomerular types as denominator to offset the number "
                                        "of glomeruli sampled per biopsy.")}
    write_provenance("14_kpmp", [S13 / f for f in ("gene_effects_by_unit.tsv.gz", "gene_effects_meta.tsv.gz",
                                                   "celltype_enrichment.tsv", "celltype_markers.tsv")]
                     + [r["pb"].cdir / "source.json" for r in runs.values()], outs, CFG["seed"], extra)


if __name__ == "__main__":
    main()
