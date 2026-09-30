"""KPMP h5ad（CELLxGENE schema）读取与 供体 × 细胞类型 pseudobulk。

- 只用 h5py：obs 列单独读取，raw/X（整数 UMI，CSR）按行块读取并用稀疏指示矩阵累加，
  不把 1.39M 细胞的矩阵整体载入。
- 缓存 <cache>/<name>/：counts.npz（行 = 供体|fine，列 = 基因符号）、rows.tsv、genes.txt、
  donors.tsv、donor_fine_counts.tsv、source.json（源文件 size / mtime / md5）。
- 分组在供体水平：组条件中的每一列在该供体（过滤后）的细胞上必须取唯一值。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from lib.repro import ROOT, md5

CHUNK = 200_000


def cache_root(kcfg: dict) -> Path:
    return Path(os.environ.get("RRG_KPMP_CACHE") or ROOT / kcfg["pseudobulk_cache"])


def _decode(a: np.ndarray) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a], dtype=object)


def read_col(h: h5py.File, grp: str, col: str) -> np.ndarray:
    """读取 obs/var 列（兼容 anndata ≥0.8 categorical 与旧式 __categories）。"""
    node = h[grp][col]
    if isinstance(node, h5py.Group):                       # categorical
        cats, codes = _decode(node["categories"][:]), node["codes"][:]
        out = np.empty(len(codes), dtype=object)
        out[codes >= 0] = cats[codes[codes >= 0]]
        out[codes < 0] = None
        return out
    vals = node[:]
    old = h[grp].get("__categories")
    if old is not None and col in old:
        cats = _decode(old[col][:])
        out = np.empty(len(vals), dtype=object)
        out[vals >= 0] = cats[vals[vals >= 0]]
        out[vals < 0] = None
        return out
    return _decode(vals) if vals.dtype.kind in "SOU" else vals


def _gene_symbols(h: h5py.File, kcfg: dict) -> np.ndarray:
    n_genes = int(h["raw/X"].attrs["shape"][1])
    for grp in ("raw/var", "var"):
        if grp in h and kcfg["gene_col"] in h[grp]:
            g = read_col(h, grp, kcfg["gene_col"])
            if len(g) != n_genes:
                raise ValueError(f"{grp}/{kcfg['gene_col']} 长度 {len(g)} 与 raw/X 列数 {n_genes} 不一致")
            return g.astype(str)
    raise KeyError(f"找不到基因名列 {kcfg['gene_col']}")


def _source_record(path: Path) -> dict:
    st = path.stat()
    return {"file": str(path), "size": st.st_size, "mtime": int(st.st_mtime)}


def _cache_valid(cdir: Path, path: Path) -> bool:
    f = cdir / "source.json"
    if not (f.is_file() and (cdir / "counts.npz").is_file()):
        return False
    rec = json.loads(f.read_text())
    cur = _source_record(path)
    return rec.get("size") == cur["size"] and rec.get("mtime") == cur["mtime"]


def _donor_table(obs: pd.DataFrame, dcfg: dict, kcfg: dict) -> pd.DataFrame:
    dcol = kcfg["donor_col"]
    group_cols = sorted({c for cond in dcfg["groups"].values() for c in cond})
    cols = list(dict.fromkeys(list(dcfg.get("meta_cols", [])) + group_cols))
    g = obs.groupby(dcol, observed=True)
    nun = g[group_cols].nunique(dropna=False)
    bad = nun[(nun > 1).any(axis=1)]
    if len(bad):
        raise ValueError(f"供体在分组列上取值不唯一：{bad.head().to_dict()}")
    don = g[cols].first()
    don.insert(0, "n_cells_total", g.size())
    spec = next((c for c in ("specimen", "SpecimenID") if c in obs), None)
    if spec:
        don.insert(1, "n_specimens", g[spec].nunique())
    for grp, cond in dcfg["groups"].items():
        ok = np.ones(len(don), bool)
        for c, vals in cond.items():
            ok &= don[c].isin(vals).values
        don[grp] = ok
    return don.rename_axis("donor").reset_index()


def build_pseudobulk(name: str, dcfg: dict, kcfg: dict) -> Path:
    path = ROOT / dcfg["file"]
    cdir = cache_root(kcfg) / name
    if _cache_valid(cdir, path):
        return cdir
    cdir.mkdir(parents=True, exist_ok=True)
    print(f"[kpmp] 构建 {name} pseudobulk 缓存 → {cdir}", flush=True)
    with h5py.File(path, "r") as h:
        need = {kcfg["donor_col"], kcfg["fine_col"], kcfg["coarse_col"], *dcfg.get("cell_filter", {}),
                *dcfg.get("meta_cols", []), *(c for cond in dcfg["groups"].values() for c in cond)}
        for c in ("specimen", "SpecimenID"):
            if c in h["obs"]:
                need.add(c)
        obs = pd.DataFrame({c: read_col(h, "obs", c) for c in sorted(need)})
        keep = obs[kcfg["fine_col"]].notna().values & obs[kcfg["donor_col"]].notna().values
        for c, vals in (dcfg.get("cell_filter") or {}).items():
            keep &= obs[c].isin(vals).values
        genes = _gene_symbols(h, kcfg)
        obs_k = obs[keep]
        key = obs_k[kcfg["donor_col"]].astype(str) + "|" + obs_k[kcfg["fine_col"]].astype(str)
        codes, uniq = pd.factorize(key, sort=True)
        cell_code = np.full(len(obs), -1, np.int64)
        cell_code[keep] = codes
        X = h["raw/X"]
        indptr = X["indptr"][:]
        n_cells, n_genes = (int(v) for v in X.attrs["shape"])
        acc = np.zeros((len(uniq), n_genes), np.float64)
        for s in range(0, n_cells, CHUNK):
            e = min(n_cells, s + CHUNK)
            cc = cell_code[s:e]
            if (cc >= 0).sum() == 0:
                continue
            a, b = indptr[s], indptr[e]
            blk = sp.csr_matrix((X["data"][a:b], X["indices"][a:b], indptr[s:e + 1] - a), shape=(e - s, n_genes))
            rows = np.flatnonzero(cc >= 0)
            G = sp.csr_matrix((np.ones(len(rows)), (cc[rows], rows)), shape=(len(uniq), e - s))
            acc += (G @ blk).toarray()
            print(f"  {e}/{n_cells}", flush=True)
    # 重复基因符号求和
    gs = pd.Index(genes)
    acc = sp.csr_matrix(acc)
    if gs.duplicated().any():
        codes_g, ug = pd.factorize(gs)
        M = sp.csr_matrix((np.ones(len(gs)), (np.arange(len(gs)), codes_g)), shape=(len(gs), len(ug)))
        acc = (acc @ M).tocsr()
        gs = pd.Index(ug)
    sp.save_npz(cdir / "counts.npz", acc)
    (cdir / "genes.txt").write_text("\n".join(gs) + "\n")
    rows = pd.DataFrame({"key": uniq})
    rows[["donor", "fine"]] = rows.key.str.split("|", n=1, expand=True)
    rows["n_cells"] = np.bincount(codes, minlength=len(uniq))
    rows.to_csv(cdir / "rows.tsv", sep="\t", index=False)
    obs_k.assign(_fine=obs_k[kcfg["fine_col"]].astype(str)).pivot_table(
        index=kcfg["donor_col"], columns="_fine", aggfunc="size", fill_value=0, observed=True
    ).rename_axis("donor").to_csv(cdir / "donor_fine_counts.tsv", sep="\t")
    _donor_table(obs_k, dcfg, kcfg).to_csv(cdir / "donors.tsv", sep="\t", index=False)
    rec = _source_record(path) | {"md5": md5(path), "n_cells_kept": int(keep.sum()), "n_cells_file": int(n_cells),
                                  "cell_filter": dcfg.get("cell_filter") or {}}
    (cdir / "source.json").write_text(json.dumps(rec, indent=1))
    return cdir


class Pseudobulk:
    """供体 × fine 的计数；coarse() 按映射求和。"""

    def __init__(self, name: str, dcfg: dict, kcfg: dict):
        cdir = build_pseudobulk(name, dcfg, kcfg)
        self.name, self.cdir = name, cdir
        self.X = sp.load_npz(cdir / "counts.npz").tocsr()
        self.genes = pd.Index((cdir / "genes.txt").read_text().split("\n")[:-1])
        self.rows = pd.read_csv(cdir / "rows.tsv", sep="\t", keep_default_na=False)
        self.donors = pd.read_csv(cdir / "donors.tsv", sep="\t", keep_default_na=False).set_index("donor")
        self.donors.index = self.donors.index.astype(str)
        self.rows["donor"] = self.rows.donor.astype(str)
        self.fine_counts = pd.read_csv(cdir / "donor_fine_counts.tsv", sep="\t", index_col=0)
        self.fine_counts.index = self.fine_counts.index.astype(str)
        self.source = json.loads((cdir / "source.json").read_text())
        self.groups = list(dcfg["groups"])

    def group_donors(self, grp: str) -> pd.Index:
        return self.donors.index[self.donors[grp].astype(str).str.lower() == "true"]

    def unmapped(self, mapping: dict) -> pd.DataFrame:
        known = {f for v in mapping.values() for f in v}
        n = self.rows.groupby("fine").n_cells.sum()
        return n[~n.index.isin(known)].rename("n_cells").rename_axis("fine").reset_index()

    def aggregate(self, mapping: dict) -> tuple[pd.DataFrame, sp.csr_matrix]:
        """mapping：新标签 → fine 名单。返回 (rows: donor, cell_type, n_cells, key) 与计数矩阵。"""
        lab = {f: k for k, v in mapping.items() for f in v}
        r = self.rows.assign(cell_type=self.rows.fine.map(lab))
        r = r[r.cell_type.notna()]
        key = r.donor + "|" + r.cell_type
        codes, uniq = pd.factorize(key, sort=True)
        G = sp.csr_matrix((np.ones(len(r)), (codes, r.index.values)), shape=(len(uniq), len(self.rows)))
        out = pd.DataFrame({"key": uniq})
        out[["donor", "cell_type"]] = out.key.str.split("|", n=1, expand=True)
        out["n_cells"] = np.bincount(codes, weights=r.n_cells.values, minlength=len(uniq)).astype(int)
        return out, (G @ self.X).tocsr()

    def fine_level(self, names: list[str]) -> tuple[pd.DataFrame, sp.csr_matrix]:
        return self.aggregate({f: [f] for f in names if f in set(self.rows.fine)})
