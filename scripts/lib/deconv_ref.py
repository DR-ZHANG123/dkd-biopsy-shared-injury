"""Stage 16 反卷积的输入构建：KPMP 细胞类型 / 状态参照与 bulk 混合物（线性尺度整数计数）。

参照：stage 14 的 KPMP pseudobulk 缓存（供体 × SubclassLevel2 计数）。每一行 = 一个供体的一个细胞状态，
type = config deconv.types[区室] 的粗类，state = KPMP 细粒度注释（覆盖健康、CKD、AKI 供体的全部状态，
如 POD / dPOD、PT-S1 / aPT / dPT）。BayesPrism 的 φ 只取决于每个状态内计数之和的归一化，
因此用供体级 pseudobulk 行代替单细胞行不改变 φ；get.exp.stat 的差异检验以供体行为重复单位。

解离偏倚：BayesPrism 的 θ 先验是均匀 Dirichlet，不使用参照里的细胞比例，
所以 scRNA 足细胞极少（解离损失）只影响 POD 状态 φ 的估计精度，不会把比例先验压低；
snRNA 为主参照（足细胞 / 系膜捕获充分），scRNA 作敏感性。核内 vs 全细胞转录本的逐基因差异由
BayesPrism 参照更新步骤（跨样本共享的逐基因平台偏移）吸收。

混合物：每个 (series, 区室) 单独一批（参照更新在批内共享，使平台效应不跨 series 混用）。
芯片 log2 → 2^x；RNA-seq log2(CPM+1) → 2^x − 1；每样本缩放到 target_total 后取整。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from lib.kpmp import Pseudobulk
from lib.repro import ROOT, load_config

CFG = load_config()
DC = CFG["deconv"]
K = CFG["kpmp"]
OUT = ROOT / DC["out_dir"]
INTERIM = ROOT / CFG["paths"]["interim"]


def ref_dir(ds: str, comp: str) -> Path:
    return OUT / "reference" / f"{ds}_{comp}"


def mix_path(series: str, comp: str) -> Path:
    return OUT / "mixture" / f"{series}__{comp}.tsv.gz"


def bulk_gene_union() -> set[str]:
    g: set[str] = set()
    for f in sorted(INTERIM.glob("*_expr.parquet")):
        g |= set(pd.read_parquet(f, columns=[]).index.astype(str))
    return g


def _state_type(comp: str) -> dict[str, str]:
    out = {}
    for t, states in DC["types"][comp].items():
        for s in states:
            if s in out:
                raise ValueError(f"{comp}: 状态 {s} 同时属于 {out[s]} 与 {t}")
            out[s] = t
    return out


def _donor_group(pb: Pseudobulk) -> pd.Series:
    g = pd.Series("OTHER", index=pb.donors.index)
    for grp in ("AKI", "OTHER_CKD", "DKD", "REF"):          # 后写覆盖前写：REF 优先
        g[pb.group_donors(grp)] = grp
    return g


def build_reference(name: str, comp: str, genes_keep: set[str]) -> dict:
    """写 ref_dir/{counts_triplet.parquet（1 起始 i/j/x，行 = rows.tsv，列 = genes.txt）, rows.tsv, genes.txt}；返回摘要（状态数、未映射细胞数）。"""
    spec = DC["references"][name]
    ds = spec["dataset"]
    pb = Pseudobulk(ds, K["datasets"][ds], K)
    st = _state_type(comp)
    rows = pb.rows.copy()
    rows["type"] = rows.fine.map(st)
    unmapped = rows[rows.type.isna() & ~rows.fine.isin(DC["exclude_fine"])].groupby("fine").n_cells.sum()
    ok = rows.type.notna() & (rows.n_cells >= DC["min_cells_row"])
    n_rows = rows[ok].groupby("fine").size()
    good_states = n_rows.index[n_rows >= DC["min_state_rows"]]
    ok &= rows.fine.isin(good_states)
    if spec.get("drop_state_regex"):
        ok &= ~rows.fine.str.contains(spec["drop_state_regex"], regex=True)
    r = rows[ok].copy()
    gmask = pb.genes.isin(genes_keep)
    X = pb.X[r.index.values][:, np.flatnonzero(gmask)]
    detected = np.asarray((X > 0).sum(0)).ravel() > 0
    X = X[:, np.flatnonzero(detected)].tocsr()
    genes = pb.genes[gmask][detected]
    grp = _donor_group(pb)
    r["group"] = grp.reindex(r.donor).values
    r = r.rename(columns={"fine": "state"})[["key", "donor", "group", "type", "state", "n_cells"]]
    d = ref_dir(name, comp)
    d.mkdir(parents=True, exist_ok=True)
    for stale in ("markers.txt", "exp_stat.rds"):         # R 端按参照缓存的标志基因；参照重建后失效
        (d / stale).unlink(missing_ok=True)
    C = X.tocoo()
    pd.DataFrame({"i": C.row.astype(np.int32) + 1, "j": C.col.astype(np.int32) + 1,
                  "x": C.data.astype(np.float64)}).to_parquet(d / "counts_triplet.parquet")
    r.to_csv(d / "rows.tsv", sep="\t", index=False)
    (d / "genes.txt").write_text("\n".join(genes) + "\n")
    dropped = n_rows[n_rows < DC["min_state_rows"]]
    return {"dataset": name, "compartment": comp, "rows": r, "unmapped": unmapped,
            "dropped_states": dropped, "n_genes": len(genes)}


def reference_summary(info: dict) -> pd.DataFrame:
    r = info["rows"]
    tab = (r.groupby(["type", "state"])
           .agg(n_rows=("donor", "size"), n_cells=("n_cells", "sum"),
                n_ref_donors=("group", lambda x: int((x == "REF").sum())),
                n_ckd_donors=("group", lambda x: int(x.isin(["DKD", "OTHER_CKD"]).sum())),
                n_aki_donors=("group", lambda x: int((x == "AKI").sum())))
           .reset_index())
    tab.insert(0, "compartment", info["compartment"])
    tab.insert(0, "dataset", info["dataset"])
    tab["cell_frac_in_reference"] = tab.n_cells / tab.n_cells.sum()
    return tab


# ---------------------------------------------------------------- 混合物
def to_linear(E: pd.DataFrame, technology: str) -> pd.DataFrame:
    """log2 → 线性；芯片可选减去样本内背景分位。"""
    E = E.loc[E.notna().all(axis=1)]
    if "rna" in technology.lower():
        L = np.power(2.0, E) - 1.0
    else:
        L = np.power(2.0, E)
        q = DC["mixture"]["array_bg_quantile"]
        if q is not None:
            L = L - L.quantile(q, axis=0)
    L = L.clip(lower=0.0)
    L = L / L.sum(axis=0) * DC["mixture"]["target_total"]
    return L.round().astype(np.int64)


def mixture_groups(uids: pd.Index, samples: pd.DataFrame) -> dict[tuple[str, str], pd.Index]:
    s = samples.loc[uids]
    return {(ser, comp): g.index for (ser, comp), g in s.groupby(["series", "compartment"])}


def write_mixture(series: str, comp: str, uids: pd.Index, samples: pd.DataFrame) -> Path:
    E = pd.read_parquet(INTERIM / f"{series}_expr.parquet")
    gsm = samples.loc[uids, "gsm"].astype(str)
    E = E[gsm.values]
    E.columns = uids
    tech = str(samples.loc[uids[0], "technology"])
    L = to_linear(E, tech)
    L = L.loc[L.sum(axis=1) > 0]
    p = mix_path(series, comp)
    p.parent.mkdir(parents=True, exist_ok=True)
    L.rename_axis("gene").to_csv(p, sep="\t")
    return p
