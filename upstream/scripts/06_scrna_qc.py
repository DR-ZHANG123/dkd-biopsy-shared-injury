#!/usr/bin/env python
"""Quality control of the two single-cell datasets, each processed independently.

Cell calling from the barcode-rank curve, then gene-count and mitochondrial
filters, then doublet removal. The two datasets are never merged before QC."""
from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
SC_RAW = ROOT / "data_raw" / "scrna"
META = ROOT / "data_raw" / "metadata"
OUT = ROOT / "data_processed" / "scrna"

SEED = 20260722
sc.settings.verbosity = 1

DATASETS = {
    "GSE131882": dict(
        path=SC_RAW / "GSE131882" / "mtx",
        assay="snRNA",
        id_type="ensembl",
        samples={"control.s1": "Control", "control.s2": "Control",
                 "control.s3": "Control", "diabetes.s1": "DKD",
                 "diabetes.s2": "DKD", "diabetes.s3": "DKD"},
        max_mito=5.0,
        min_counts=500,
    ),
    "GSE209781": dict(
        path=SC_RAW / "GSE209781" / "raw",
        assay="scRNA",
        id_type="symbol",
        samples={"NM01": "Control", "NM02": "Control", "NM03": "Control",
                 "DKD01": "DKD", "DKD02": "DKD", "DKD03": "DKD"},
        max_mito=25.0,
        min_counts=500,
    ),
}


def ensembl_to_symbol() -> dict[str, str]:
    out = {}
    with gzip.open(META / "Homo_sapiens.gene_info.gz", "rt") as fh:
        next(fh)
        for line in fh:
            f = line.split("\t")
            if f[0] != "9606":
                continue
            sym = f[10] if f[10] != "-" else f[2]
            for x in f[5].split("|"):
                if x.startswith("Ensembl:"):
                    out[x.split(":", 1)[1]] = sym
    return out


def knee_threshold(counts: np.ndarray, floor: int = 500) -> float:
    """Knee point of the barcode-rank curve, used as the cell-calling threshold."""
    c = np.sort(counts[counts >= floor])[::-1]
    if c.size < 100:
        return float(floor)
    x = np.log10(np.arange(1, c.size + 1))
    y = np.log10(c)
    win = max(11, (c.size // 100) | 1)
    ys = np.convolve(y, np.ones(win) / win, mode="same")
    d2 = np.gradient(np.gradient(ys, x), x)
    lo, hi = int(0.02 * c.size), int(0.98 * c.size)
    idx = lo + int(np.argmin(d2[lo:hi]))
    return float(max(floor, c[idx]))


def read_mtx_dir(d: Path, id_type: str) -> ad.AnnData:
    """Read a 10x matrix directory into an AnnData object."""
    feats = pd.read_csv(d / "features.tsv.gz", sep="\t", header=None)
    if feats.shape[1] >= 2 and id_type == "symbol":
        return sc.read_10x_mtx(d, var_names="gene_symbols", cache=False)
    if feats.shape[1] >= 2:
        return sc.read_10x_mtx(d, var_names="gene_ids", cache=False)
    import scipy.io as sio
    with gzip.open(d / "matrix.mtx.gz", "rb") as fh:
        mat = sio.mmread(fh).T.tocsr()
    bcs = pd.read_csv(d / "barcodes.tsv.gz", sep="\t", header=None)[0]
    a = ad.AnnData(mat.astype(np.float32))
    a.var_names = feats[0].astype(str).values
    a.obs_names = bcs.astype(str).values
    return a


def load_sample(ds: str, cfg: dict, sample: str) -> ad.AnnData:
    a = read_mtx_dir(cfg["path"] / sample, cfg["id_type"])
    a.var_names_make_unique()
    a.obs["sample"] = sample
    a.obs["group"] = cfg["samples"][sample]
    a.obs["dataset"] = ds
    a.obs_names = [f"{ds}|{sample}|{b}" for b in a.obs_names]
    return a


def qc_dataset(ds: str, cfg: dict, e2s: dict[str, str]) -> ad.AnnData:
    print(f"\n===== {ds} ({cfg['assay']}) =====")
    parts, stats = [], []
    for sample in cfg["samples"]:
        a = load_sample(ds, cfg, sample)
        n0 = a.n_obs
        total = np.asarray(a.X.sum(axis=1)).ravel()
        thr = knee_threshold(total, cfg["min_counts"])
        a = a[total >= thr].copy()
        print(f"  [{sample}] barcodes {n0} -> {a.n_obs} (UMI threshold {thr:.0f})")
        stats.append(dict(dataset=ds, sample=sample, group=cfg["samples"][sample],
                          barcodes_raw=n0, umi_threshold=thr,
                          cells_called=a.n_obs))
        parts.append(a)

    adata = ad.concat(parts, join="outer", index_unique=None)
    adata.X = adata.X.astype(np.float32)

    if cfg["id_type"] == "ensembl":
        sym = pd.Series(adata.var_names).map(e2s)
        adata.var["ensembl"] = adata.var_names
        adata = adata[:, sym.notna().values].copy()
        adata.var["symbol"] = sym.dropna().values
        adata.var_names = adata.var["symbol"].astype(str).values
        adata.var_names_make_unique()

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.match(r"^RP[SL]")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"],
                               percent_top=None, log1p=False, inplace=True)

    n_before = adata.n_obs
    keep = ((adata.obs.n_genes_by_counts >= 200) &
            (adata.obs.n_genes_by_counts <= np.percentile(
                adata.obs.n_genes_by_counts, 99)) &
            (adata.obs.pct_counts_mt <= cfg["max_mito"]))
    adata = adata[keep].copy()
    print(f"  QC filter: {n_before} -> {adata.n_obs} cells "
          f"(gene count 200 to P99, mito <= {cfg['max_mito']}%)")

    sc.pp.scrublet(adata, batch_key="sample", random_state=SEED)
    n_before = adata.n_obs
    adata = adata[~adata.obs.predicted_doublet].copy()
    print(f"  doublet removal: {n_before} -> {adata.n_obs} cells")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")
    sc.pp.pca(adata, n_comps=50, mask_var="highly_variable", random_state=SEED)
    import harmonypy
    ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, ["sample"],
                               max_iter_harmony=20)
    z = np.asarray(ho.Z_corr)
    adata.obsm["X_pca_harmony"] = z if z.shape[0] == adata.n_obs else z.T
    sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=15,
                    random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden",
                 flavor="igraph", n_iterations=2, random_state=SEED)
    print(f"  leiden clusters: {adata.obs.leiden.nunique()}")

    pd.DataFrame(stats).to_csv(
        ROOT / "results" / "tables" / f"T03_scqc_{ds}.tsv",
        sep="\t", index=False)
    return adata


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e2s = ensembl_to_symbol()
    print(f"[annot] Ensembl to Symbol: {len(e2s)} entries")
    for ds, cfg in DATASETS.items():
        adata = qc_dataset(ds, cfg, e2s)
        out = OUT / f"{ds}_qc.h5ad"
        adata.write_h5ad(out, compression="gzip")
        print(f"[write] {out}  {adata.shape}")


if __name__ == "__main__":
    main()
