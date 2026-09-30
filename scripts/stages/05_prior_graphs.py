"""Stage 05：先验基因图（与任何 bulk / DKD 标签无关）。

1. STRING v12 PPI（combined_score ≥ graphs.string_min_score）→ edges_ppi.tsv
2. 对照供体单核共表达：每个数据集只取 group == Control 的细胞，在 供体 × 细胞类型 内
   按对照细胞重算的 PCA 空间做 KMeans 聚成 metacell（~coexpr_cells_per_metacell 个细胞），
   counts 求和 → CP10K → log1p；两个数据集分别算跨 metacell 的 Spearman，Fisher z 平均，
   每个基因取 |r| 最高的 top-k 作无向边 → edges_coexpr.tsv
3. B4 对照：保度数双边交换重连（边权随边移动），seed 0..n-1
4. graph_summary.tsv + PROVENANCE.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "05_graphs"


# --------------------------------------------------------------------------- utils
def canon_edges(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """无向去重：按字典序排 (a,b)，同一对保留最大权重，去自环。"""
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    lo = np.where(a < b, a, b)
    hi = np.where(a < b, b, a)
    df = pd.DataFrame({"gene_a": lo, "gene_b": hi, "weight": np.asarray(w, dtype=float)})
    df = df[df.gene_a != df.gene_b]
    df = df.groupby(["gene_a", "gene_b"], as_index=False, sort=True)["weight"].max()
    return df


def edge_keys(df: pd.DataFrame) -> set:
    return set(zip(df.gene_a, df.gene_b))


def write_edges(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, float_format="%.4f")


# --------------------------------------------------------------------------- PPI
def build_ppi(cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw = ROOT / cfg["paths"]["raw"] / "string"
    ver = cfg["graphs"]["string_version"]
    info = pd.read_csv(raw / f"9606.protein.info.v{ver}.txt.gz", sep="\t",
                       usecols=[0, 1], names=["pid", "symbol"], header=0)
    pid2sym = dict(zip(info.pid, info.symbol))
    links = pd.read_csv(raw / f"9606.protein.links.v{ver}.txt.gz", sep=" ")
    n_raw = len(links)
    links = links[links.combined_score >= cfg["graphs"]["string_min_score"]]
    a = links.protein1.map(pid2sym)
    b = links.protein2.map(pid2sym)
    ok = a.notna() & b.notna()
    df = canon_edges(a[ok].values, b[ok].values, links.combined_score[ok].values / 1000.0)
    meta = {"string_links_total": int(n_raw), "string_links_ge_threshold": int(len(links)),
            "string_unmapped_links": int((~ok).sum()), "ppi_edges": int(len(df))}
    return df, meta


# --------------------------------------------------------------------------- metacells
def control_metacells(h5ad: Path, gcfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (metacell × gene 的 log CP10K 表达, metacell 元数据)。只读对照细胞。"""
    import anndata as ad
    import scanpy as sc
    from scipy import sparse
    from sklearn.cluster import KMeans

    col_d, col_g, col_ct = gcfg["sc_donor_col"], gcfg["sc_group_col"], gcfg["sc_celltype_col"]
    adata = ad.read_h5ad(h5ad)
    keep = (adata.obs[col_g].astype(str) == gcfg["sc_control_value"]).values
    keep &= ~adata.obs[col_ct].astype(str).isin(gcfg["coexpr_exclude_celltypes"]).values
    adata = adata[keep].copy()
    del adata.raw
    counts = adata.layers["counts"]
    counts = sparse.csr_matrix(counts)

    # 仅用对照细胞重算 PCA（不复用在全体细胞上拟合的 X_pca）
    tmp = ad.AnnData(X=counts.astype(np.float32), obs=adata.obs[[col_d, col_ct]].copy(),
                     var=pd.DataFrame(index=adata.var_names))
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=gcfg["coexpr_n_hvg"], batch_key=col_d)
    hv = tmp[:, tmp.var.highly_variable].copy()
    sc.pp.scale(hv, max_value=10)
    sc.tl.pca(hv, n_comps=gcfg["coexpr_n_pcs"], random_state=seed)
    pcs = hv.obsm["X_pca"]
    del tmp, hv

    size = gcfg["coexpr_cells_per_metacell"]
    cap = gcfg["coexpr_max_metacells_per_group"]
    min_cells = gcfg["coexpr_min_cells_per_group"]
    labels = np.full(adata.n_obs, -1, dtype=np.int64)
    rows = []
    nxt = 0
    obs = adata.obs
    for (donor, ct), idx in obs.groupby([col_d, col_ct], observed=True).indices.items():
        n = len(idx)
        if n < min_cells:
            continue
        k = int(min(max(n // size, 1), cap))
        if k == 1:
            lab = np.zeros(n, dtype=np.int64)
        else:
            lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(pcs[idx])
        for j in range(k):
            rows.append({"metacell": nxt + j, "donor": str(donor), "cell_type": str(ct),
                         "n_cells": int((lab == j).sum())})
        labels[idx] = nxt + lab
        nxt += k
    meta = pd.DataFrame(rows)
    sel = labels >= 0
    agg = sparse.csr_matrix((np.ones(sel.sum()), (labels[sel], np.where(sel)[0])),
                            shape=(nxt, adata.n_obs))
    mc = np.asarray((agg @ counts).todense(), dtype=np.float64)
    lib = mc.sum(axis=1, keepdims=True)
    expr = np.log1p(mc / lib * 1e4).astype(np.float32)
    df = pd.DataFrame(expr, columns=adata.var_names.astype(str))
    df = df.T.groupby(level=0).max().T  # 重复 symbol 取最大（极少）
    return df, meta


def spearman_z(expr: np.ndarray) -> np.ndarray:
    """列 = 基因。返回 Fisher z(Spearman r) 的基因 × 基因矩阵（float32）。"""
    from scipy.stats import rankdata
    r = rankdata(expr, axis=0).astype(np.float64)
    r -= r.mean(axis=0, keepdims=True)
    r /= np.linalg.norm(r, axis=0, keepdims=True)
    c = (r.T @ r).astype(np.float32)
    np.clip(c, -0.999999, 0.999999, out=c)
    return np.arctanh(c)


def build_coexpr(cfg: dict, seed: int) -> tuple[pd.DataFrame, dict, list[Path]]:
    g = cfg["graphs"]
    base = ROOT / cfg["old_project_processed"] / "scrna"
    files = [base / f"{d}_annotated.h5ad" for d in g["coexpr_datasets"]]
    mats, info = {}, {}
    for d, f in zip(g["coexpr_datasets"], files):
        df, meta = control_metacells(f, g, seed)
        mats[d] = df
        info[d] = {"control_donors": sorted(meta.donor.unique().tolist()),
                   "n_control_donors": int(meta.donor.nunique()),
                   "n_metacells": int(len(meta)),
                   "n_cells_used": int(meta.n_cells.sum()),
                   "cells_per_metacell_median": float(meta.n_cells.median()),
                   "metacells_per_celltype": meta.groupby("cell_type").size().to_dict()}
        meta.to_csv(ROOT / "results" / STAGE / f"metacells_{d}.tsv", sep="\t", index=False)
        print(f"[coexpr] {d}: {info[d]['n_control_donors']} donors, {len(meta)} metacells", flush=True)
    common = sorted(set.intersection(*[set(m.columns) for m in mats.values()]))
    frac = g["coexpr_min_metacell_frac"]
    genes = [x for x in common if all((mats[d][x].values > 0).mean() >= frac for d in mats)]
    info["n_genes_common"] = len(common)
    info["n_genes_expressed"] = len(genes)
    print(f"[coexpr] genes common={len(common)} expressed={len(genes)}", flush=True)
    zsum = None
    for d in mats:
        z = spearman_z(mats[d][genes].values)
        zsum = z if zsum is None else zsum + z
        del z
    zsum /= len(mats)
    rbar = np.abs(np.tanh(zsum))
    np.fill_diagonal(rbar, -1.0)
    k = g["coexpr_k"]
    top = np.argpartition(-rbar, k, axis=1)[:, :k]
    gi = np.repeat(np.arange(len(genes)), k)
    gj = top.ravel()
    w = rbar[gi, gj]
    gn = np.array(genes, dtype=object)
    df = canon_edges(gn[gi], gn[gj], w)
    info["coexpr_edges"] = int(len(df))
    info["coexpr_weight_min"] = float(df.weight.min())
    info["coexpr_weight_median"] = float(df.weight.median())
    return df, info, files


# --------------------------------------------------------------------------- rewiring
def rewire(df: pd.DataFrame, seed: int, swap_factor: int) -> tuple[pd.DataFrame, dict]:
    """保度数双边交换：(a,b),(c,d) → (a,d),(c,b)，拒绝自环与重边；边权随边移动。"""
    nodes = pd.Index(sorted(set(df.gene_a) | set(df.gene_b)))
    u = nodes.get_indexer(df.gene_a).astype(np.int64)
    v = nodes.get_indexer(df.gene_b).astype(np.int64)
    w = df.weight.values.copy()
    m = len(u)
    n = len(nodes)
    key = lambda x, y: min(x, y) * n + max(x, y)  # noqa: E731
    present = set((np.minimum(u, v) * n + np.maximum(u, v)).tolist())
    rng = np.random.default_rng(seed)
    target = swap_factor * m
    done, tries, max_tries = 0, 0, 20 * target
    block = 1 << 20
    while done < target and tries < max_tries:
        e1s = rng.integers(0, m, block)
        e2s = rng.integers(0, m, block)
        flips = rng.random(block) < 0.5
        for e1, e2, fl in zip(e1s.tolist(), e2s.tolist(), flips.tolist()):
            tries += 1
            if e1 == e2:
                continue
            a, b = u[e1], v[e1]
            c, d = (v[e2], u[e2]) if fl else (u[e2], v[e2])
            if a == d or c == b or a == c or b == d:
                continue
            k1, k2 = key(a, d), key(c, b)
            if k1 in present or k2 in present:
                continue
            present.discard(key(a, b))
            present.discard(key(c, d))
            present.add(k1)
            present.add(k2)
            u[e1], v[e1] = a, d
            u[e2], v[e2] = c, b
            done += 1
            if done >= target:
                break
    out = canon_edges(nodes.values[u], nodes.values[v], w)
    return out, {"swaps_done": int(done), "swaps_target": int(target), "tries": int(tries)}


# --------------------------------------------------------------------------- summary
def degree_series(df: pd.DataFrame) -> pd.Series:
    return pd.concat([df.gene_a, df.gene_b]).value_counts()


def jaccard(s1: set, s2: set) -> float:
    return len(s1 & s2) / max(len(s1 | s2), 1)


def summarize(graphs: dict[str, pd.DataFrame], originals: dict[str, set],
              node_genes: list[str] | None) -> pd.DataFrame:
    rows = []
    for name, df in graphs.items():
        deg = degree_series(df)
        base = name.split("_rewired")[0]
        row = {"graph": name, "n_nodes": int(len(deg)), "n_edges": int(len(df)),
               "degree_median": float(deg.median()), "degree_mean": float(deg.mean()),
               "degree_max": int(deg.max()), "weight_median": float(df.weight.median()),
               "jaccard_vs_original": jaccard(edge_keys(df), originals[base])}
        if node_genes is not None:
            ns = set(node_genes)
            sub = df[df.gene_a.isin(ns) & df.gene_b.isin(ns)]
            sdeg = degree_series(sub)
            row["node_genes_in_graph"] = int(len(ns & set(deg.index)))
            row["node_coverage_induced"] = float(len(sdeg) / len(ns))
            row["induced_edges"] = int(len(sub))
            row["induced_degree_median"] = float(sdeg.reindex(node_genes).fillna(0).median())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_global_seed(seed)
    g = cfg["graphs"]
    assert g["coexpr_controls_only"], "共表达图只允许用对照供体构建"
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)

    ppi, ppi_meta = build_ppi(cfg)
    write_edges(ppi, out / "edges_ppi.tsv")
    print(f"[ppi] {ppi_meta}", flush=True)

    coexpr, co_meta, sc_files = build_coexpr(cfg, seed)
    write_edges(coexpr, out / "edges_coexpr.tsv")

    graphs = {"ppi": ppi, "coexpr": coexpr}
    originals = {k: edge_keys(v) for k, v in graphs.items()}
    rew_meta = {}
    outputs = [out / "edges_ppi.tsv", out / "edges_coexpr.tsv"]
    for name in ("ppi", "coexpr"):
        for k in range(g["rewire_n_seeds"]):
            r, m = rewire(graphs[name], k, g["rewire_swap_factor"])
            p = out / f"edges_{name}_rewired_seed{k}.tsv"
            write_edges(r, p)
            outputs.append(p)
            graphs[f"{name}_rewired_seed{k}"] = r
            rew_meta[f"{name}_seed{k}"] = m
            print(f"[rewire] {name} seed{k} {m}", flush=True)

    genes_txt = ROOT / cfg["paths"]["processed"] / "genes.txt"
    node_genes = genes_txt.read_text().split() if genes_txt.is_file() else None
    summ = summarize(graphs, originals, node_genes)
    summ["jaccard_ppi_vs_coexpr"] = jaccard(originals["ppi"], originals["coexpr"])
    summ.to_csv(out / "graph_summary.tsv", sep="\t", index=False, float_format="%.4f")
    outputs += [out / "graph_summary.tsv"] + [out / f"metacells_{d}.tsv" for d in g["coexpr_datasets"]]
    print(summ.to_string(), flush=True)

    ver = g["string_version"]
    inputs = [ROOT / cfg["paths"]["raw"] / "string" / f"9606.protein.{t}.v{ver}.txt.gz"
              for t in ("links", "info")] + sc_files
    if node_genes is not None:
        inputs.append(genes_txt)
    write_provenance(STAGE, inputs, outputs, seed,
                     extra={"ppi": ppi_meta, "coexpr": co_meta, "rewire": rew_meta,
                            "node_genes_available": node_genes is not None,
                            "jaccard_ppi_vs_coexpr": jaccard(originals["ppi"], originals["coexpr"])})


if __name__ == "__main__":
    main()
