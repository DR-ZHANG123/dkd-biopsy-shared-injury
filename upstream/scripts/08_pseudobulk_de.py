#!/usr/bin/env python
"""Donor-level pseudo-bulk differential expression per cell type.

Counts are summed per donor and cell type and tested with DESeq2. Cells are never
treated as replicates, which would inflate the false discovery rate. With three
donors per group few genes survive a 5 percent FDR, so cross-dataset direction
agreement carries the stability requirement instead."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data_processed" / "scrna"
TAB = ROOT / "results" / "tables"

MIN_CELLS_PER_DONOR = 20
MIN_DONORS_PER_GROUP = 2
MIN_COUNT = 10
FDR = 0.05


def build_pseudobulk(adata: sc.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum raw counts per donor and cell type, dropping cell types below the minimum cell count."""
    counts = adata.layers["counts"]
    genes = adata.var_names
    keys, mats, meta = [], [], []
    grp = adata.obs.groupby(["sample", "cell_type"], observed=True).indices
    for (sample, ct), idx in grp.items():
        if ct == "Unassigned" or len(idx) < MIN_CELLS_PER_DONOR:
            continue
        vec = np.asarray(counts[idx].sum(axis=0)).ravel()
        keys.append(f"{sample}|{ct}")
        mats.append(vec)
        meta.append(dict(key=f"{sample}|{ct}", sample=sample, cell_type=ct,
                         group=adata.obs.loc[adata.obs["sample"] == sample,
                                             "group"].iloc[0],
                         n_cells=len(idx)))
    pb = pd.DataFrame(np.vstack(mats).T, index=genes, columns=keys)
    return pb, pd.DataFrame(meta).set_index("key")


def run_deseq(counts: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    dds = DeseqDataSet(counts=counts.T.astype(int), metadata=meta,
                       design="~group", refit_cooks=True, quiet=True)
    dds.deseq2()
    stat = DeseqStats(dds, contrast=["group", "DKD", "Control"], quiet=True)
    stat.summary()
    res = stat.results_df.copy()
    res.index.name = "gene"
    return res.reset_index()


def analyse(ds: str) -> pd.DataFrame:
    adata = sc.read_h5ad(PROC / f"{ds}_annotated.h5ad")
    pb, meta = build_pseudobulk(adata)
    pb.to_csv(PROC / f"pseudobulk_{ds}.tsv.gz", sep="\t")
    meta.to_csv(PROC / f"pseudobulk_{ds}_meta.tsv", sep="\t")
    print(f"\n===== {ds}: {pb.shape[1]} pseudo-bulk samples (donor x cell type) =====")

    out = []
    for ct, sub in meta.groupby("cell_type"):
        n_dkd = (sub.group == "DKD").sum()
        n_ctl = (sub.group == "Control").sum()
        if min(n_dkd, n_ctl) < MIN_DONORS_PER_GROUP:
            print(f"  [skip] {ct}: DKD={n_dkd} Ctrl={n_ctl}, too few donors")
            continue
        cnt = pb[sub.index]
        cnt = cnt[cnt.sum(axis=1) >= MIN_COUNT]
        res = run_deseq(cnt, sub[["group"]])
        res["cell_type"] = ct
        res["n_donor_DKD"] = n_dkd
        res["n_donor_Control"] = n_ctl
        out.append(res)
        sig = (res.padj < FDR).sum()
        print(f"  [{ct}] donors {n_dkd}/{n_ctl}, {cnt.shape[0]} genes, "
              f"significant at FDR<{FDR}: {sig}")

    de = pd.concat(out, ignore_index=True)
    de["dataset"] = ds
    de.to_csv(TAB / f"T06_pseudobulk_de_{ds}.tsv.gz", sep="\t", index=False)
    return de


def replication(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of donor-level log2 fold changes between the two datasets."""
    rows = []
    common_ct = set(a.cell_type) & set(b.cell_type)
    for ct in sorted(common_ct):
        x = a[(a.cell_type == ct) & (a.padj < FDR)][["gene", "log2FoldChange"]]
        y = b[b.cell_type == ct][["gene", "log2FoldChange", "pvalue"]]
        m = x.merge(y, on="gene", suffixes=("_disc", "_rep")).dropna()
        if len(m) < 10:
            continue
        same = np.sign(m.log2FoldChange_disc) == np.sign(m.log2FoldChange_rep)
        from scipy.stats import binomtest
        p = binomtest(int(same.sum()), len(m), 0.5, alternative="greater").pvalue
        rows.append(dict(cell_type=ct, n_sig_discovery=len(x),
                         n_testable=len(m),
                         concordance=round(float(same.mean()), 3),
                         binom_p=float(p)))
    return pd.DataFrame(rows)


def main() -> None:
    de1 = analyse("GSE131882")
    de2 = analyse("GSE209781")
    rep = replication(de1, de2)
    rep.to_csv(TAB / "T07_de_replication.tsv", sep="\t", index=False)
    print("\ncross-dataset direction agreement (GSE131882 discovery -> GSE209781 validation):")
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
