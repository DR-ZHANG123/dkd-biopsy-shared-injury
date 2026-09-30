"""模型线（stage 06–09）的数据读取：只按 notes/02_interfaces.md 的接口文件工作。

- 预训练相关函数（load_samples_unlabelled / pretrain_uids / load_edges / variant_edges）不读取 diagnosis。
- 下游函数（downstream_split）才读 diagnosis，只在 stage 06/08/09 调用。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lib.repro import ROOT

PROC = ROOT / "data" / "processed"
GRAPHS = ROOT / "results" / "05_graphs"
SAMPLE_COLS_NODIAG = ["sample_uid", "series", "gse", "gsm", "technology", "compartment", "role",
                      "dup_group", "keep_for_pretrain"]


@dataclass(frozen=True)
class Fold:
    task: str
    compartment: str
    fold: str
    test_cohort: str

    @property
    def key(self) -> str:
        return f"{self.task}_{self.compartment}_{self.fold}"


# ---------------------------------------------------------------- 基础读取
def load_genes() -> list[str]:
    return [g.strip() for g in (PROC / "genes.txt").read_text().splitlines() if g.strip()]


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def load_samples_unlabelled() -> pd.DataFrame:
    """samples.tsv 去掉 diagnosis 列（stage 07 与 PCA 语料构造只用这个）。"""
    df = pd.read_csv(PROC / "samples.tsv", sep="\t", dtype=str,
                     usecols=lambda c: c != "diagnosis")
    df["keep_for_pretrain"] = _as_bool(df["keep_for_pretrain"])
    df["dup_group"] = df["dup_group"].fillna(df["sample_uid"])
    df = df.set_index("sample_uid", drop=False)
    df["cohort"] = load_cohort_map(df)
    return df


def load_diagnosis() -> pd.Series:
    """仅供下游评估（stage 06/08/09）。"""
    df = pd.read_csv(PROC / "samples.tsv", sep="\t", dtype=str, usecols=["sample_uid", "diagnosis"])
    return df.set_index("sample_uid")["diagnosis"]


def load_cohort_map(samples: pd.DataFrame) -> pd.Series:
    """sample_uid → cohort。优先用 stage 04 的 cohorts.tsv（ERCB 按芯片批次跨 series 合并为一个 cohort）；
    cohorts.tsv 未覆盖的样本（非代表样本、预训练语料）退回 series 名。"""
    coh = samples["series"].copy()
    p = PROC / "cohorts.tsv"
    if p.exists():
        c = pd.read_csv(p, sep="\t", dtype=str).set_index("sample_uid")["cohort"]
        c = c[c.index.isin(coh.index)]
        coh.loc[c.index] = c
    return coh


def load_folds() -> pd.DataFrame:
    df = pd.read_csv(PROC / "folds.tsv", sep="\t", dtype=str)
    df["split"] = df["split"].str.strip().str.lower()
    return df


def list_folds(folds: pd.DataFrame | None = None, task: str | None = None,
               compartment: str | None = None, fold: str | None = None) -> list[Fold]:
    folds = load_folds() if folds is None else folds
    u = folds[["task", "compartment", "fold", "test_cohort"]].drop_duplicates()
    if task:
        u = u[u.task == task]
    if compartment:
        u = u[u.compartment == compartment]
    if fold:
        u = u[u.fold == str(fold)]
    return [Fold(*r) for r in u.itertuples(index=False, name=None)]


def load_ranks(uids: list[str] | None = None) -> pd.DataFrame:
    """ranks.parquet：行 sample_uid，列 genes.txt 顺序；返回 float32，列顺序强制与 genes.txt 一致。"""
    r = pd.read_parquet(PROC / "ranks.parquet")
    if r.index.name != "sample_uid" and "sample_uid" in r.columns:
        r = r.set_index("sample_uid")
    genes = load_genes()
    missing = [g for g in genes if g not in r.columns]
    if missing:
        raise ValueError(f"ranks.parquet 缺少 {len(missing)} 个 genes.txt 基因，例如 {missing[:5]}")
    r = r[genes].astype(np.float32)
    if uids is not None:
        absent = [u for u in uids if u not in r.index]
        if absent:
            raise KeyError(f"{len(absent)} 个 sample_uid 不在 ranks.parquet，例如 {absent[:3]}")
        r = r.loc[uids]
    return r


# ---------------------------------------------------------------- 预训练语料（不读 diagnosis）
def excluded_uids(fold: Fold, samples: pd.DataFrame, folds: pd.DataFrame) -> set[str]:
    """该 fold 的测试队列（整队列，含本任务不用的样本）及其 dup_group 内的一切样本。

    与标签无关：测试队列按 cohorts.tsv 的 cohort 名 / series / gse 匹配，再并上 folds.tsv 中该 fold 的 test 样本。
    同一 test_cohort 的所有 task 行都并进来，因此 T1 与 T2 同一队列的剔除集合一致。
    """
    tc = str(fold.test_cohort)
    test_rows = folds[(folds.compartment == fold.compartment) & (folds.test_cohort == tc)
                      & (folds.split == "test")]
    ex = set(test_rows.sample_uid)
    ex |= set(samples.index[(samples.cohort == tc) | (samples.series == tc) | (samples.gse == tc)])
    groups = set(samples.loc[samples.index.intersection(list(ex)), "dup_group"])
    ex |= set(samples.index[samples.dup_group.isin(groups)])
    return ex


def pretrain_uids(fold: Fold, cfg: dict, samples: pd.DataFrame | None = None,
                  folds: pd.DataFrame | None = None) -> list[str]:
    samples = load_samples_unlabelled() if samples is None else samples
    folds = load_folds() if folds is None else folds
    comps = cfg["corpus"].get(fold.compartment, [fold.compartment])
    ok = samples.keep_for_pretrain & samples.compartment.isin(comps)
    ex = excluded_uids(fold, samples, folds)
    uids = sorted(u for u in samples.index[ok] if u not in ex)
    return uids


def corpus_hash(uids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(uids)).encode()).hexdigest()[:16]


def split_pretrain_val(uids: list[str], samples: pd.DataFrame, frac: float, seed: int
                       ) -> tuple[list[str], list[str]]:
    """按 dup_group 分组随机留出 frac 作为预训练验证集（组不跨集合）。"""
    groups = samples.loc[uids, "dup_group"].to_numpy()
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    n_val = max(1, int(round(frac * len(ug))))
    val_g = set(ug[:n_val])
    val = [u for u, g in zip(uids, groups) if g in val_g]
    tr = [u for u, g in zip(uids, groups) if g not in val_g]
    return tr, val


# ---------------------------------------------------------------- 下游（读 diagnosis）
def downstream_split(fold: Fold, cfg: dict, folds: pd.DataFrame | None = None,
                     samples: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (train, test)，列：sample_uid cohort dup_group diagnosis y。只保留本任务正负类样本。"""
    folds = load_folds() if folds is None else folds
    samples = load_samples_unlabelled() if samples is None else samples
    diag = load_diagnosis()
    spec = cfg["tasks"][fold.task]
    f = folds[(folds.task == fold.task) & (folds.compartment == fold.compartment)
              & (folds.fold == fold.fold)].copy()
    f["diagnosis"] = diag.reindex(f.sample_uid).to_numpy()
    f = f[f.diagnosis.isin(spec["pos"] + spec["neg"])].copy()
    f["y"] = f.diagnosis.isin(spec["pos"]).astype(int)
    f["cohort"] = samples.loc[f.sample_uid, "cohort"].to_numpy()
    f["dup_group"] = samples.loc[f.sample_uid, "dup_group"].to_numpy()
    cols = ["sample_uid", "cohort", "dup_group", "diagnosis", "y"]
    tr = f[f.split == "train"][cols].reset_index(drop=True)
    te = f[f.split == "test"][cols].reset_index(drop=True)
    leak = set(tr.dup_group) & set(te.dup_group)
    if leak:
        raise ValueError(f"{fold.key}: {len(leak)} 个 dup_group 同时出现在 train/test")
    return tr, te


def subsample_labels(train: pd.DataFrame, frac: float, seed: int, min_per_class: int = 2) -> pd.DataFrame:
    """每个训练 cohort 内按类别分层抽取 frac 的标签样本（每类至少 min_per_class 个，不超过原有数）。"""
    if frac >= 1.0:
        return train
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in train.groupby(["cohort", "y"], sort=True):
        n = min(len(g), max(min_per_class, int(round(frac * len(g)))))
        keep.extend(rng.choice(g.index.to_numpy(), size=n, replace=False))
    return train.loc[sorted(keep)].reset_index(drop=True)


# ---------------------------------------------------------------- 图
def _edge_file(kind: str, rewired_seed: int | None) -> Path:
    name = f"edges_{kind}.tsv" if rewired_seed is None else f"edges_{kind}_rewired_seed{rewired_seed}.tsv"
    return GRAPHS / name


def load_edges(kind: str, genes: list[str], rewired_seed: int | None = None) -> torch.Tensor:
    """读 gene_a gene_b weight，裁到 genes，去自环去重，返回双向 edge_index (2, 2E)，long。"""
    df = pd.read_csv(_edge_file(kind, rewired_seed), sep="\t", usecols=["gene_a", "gene_b"], dtype=str)
    idx = {g: i for i, g in enumerate(genes)}
    a = df.gene_a.map(idx)
    b = df.gene_b.map(idx)
    ok = a.notna() & b.notna()
    a, b = a[ok].astype(np.int64).to_numpy(), b[ok].astype(np.int64).to_numpy()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    pairs = np.unique(np.stack([lo, hi], 1)[lo != hi], axis=0)
    if len(pairs) == 0:
        return torch.empty(2, 0, dtype=torch.long)
    ei = np.concatenate([pairs.T, pairs[:, ::-1].T], axis=1)
    return torch.from_numpy(np.ascontiguousarray(ei))


def rewire_induced(ei: torch.Tensor, n: int, seed: int, swap_factor: int) -> torch.Tensor:
    """在已裁到 genes.txt 的诱导子图上做保度数双边交换（networkx.double_edge_swap，拒绝自环/重边）。

    stage 05 的重连文件是在全人类图上交换的，裁到 5,000 节点后边数与度序列都不再与原诱导子图一致
    （边数显著减少），会把「图内容」与「图密度」混在一起；B4 默认改用本函数（config model_ext.rewire_mode）。
    """
    import networkx as nx
    und = ei[:, ei[0] < ei[1]].numpy().T
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(map(tuple, und))
    m = g.number_of_edges()
    if m >= 2:
        nx.double_edge_swap(g, nswap=swap_factor * m, max_tries=swap_factor * m * 20, seed=seed)
    pairs = np.array(sorted((min(a, b), max(a, b)) for a, b in g.edges()), dtype=np.int64).reshape(-1, 2)
    out = np.concatenate([pairs.T, pairs[:, ::-1].T], axis=1)
    return torch.from_numpy(np.ascontiguousarray(out))


def edge_files_for(variant: str, rewired_seed: int = 0, cfg: dict | None = None) -> list[Path]:
    kinds = {"full": ["ppi", "coexpr"], "noedge": [], "rewired": ["ppi", "coexpr"],
             "ppi_only": ["ppi"], "coexpr_only": ["coexpr"]}[variant]
    use_file = variant == "rewired" and (cfg or {}).get("model_ext", {}).get("rewire_mode", "induced") == "file"
    return [_edge_file(k, rewired_seed if use_file else None) for k in kinds]


def variant_edges(variant: str, genes: list[str], rewired_seed: int = 0, cfg: dict | None = None
                  ) -> list[torch.Tensor]:
    kinds = {"full": ["ppi", "coexpr"], "noedge": [], "rewired": ["ppi", "coexpr"],
             "ppi_only": ["ppi"], "coexpr_only": ["coexpr"]}[variant]
    if variant != "rewired":
        return [load_edges(k, genes) for k in kinds]
    mode = (cfg or {}).get("model_ext", {}).get("rewire_mode", "induced")
    if mode == "file":
        return [load_edges(k, genes, rewired_seed) for k in kinds]
    sf = (cfg or {}).get("graphs", {}).get("rewire_swap_factor", 10)
    return [rewire_induced(load_edges(k, genes), len(genes), 1000 * rewired_seed + i, sf)
            for i, k in enumerate(kinds)]


def graph_summary(edges: list[torch.Tensor], n: int) -> list[dict]:
    out = []
    for ei in edges:
        deg = torch.bincount(ei[0], minlength=n) if ei.numel() else torch.zeros(n, dtype=torch.long)
        out.append({"n_directed_edges": int(ei.shape[1]), "mean_degree": float(deg.float().mean()),
                    "frac_isolated": float((deg == 0).float().mean())})
    return out


def ranks_to_tensors(r: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """NaN → observed=False，ranks 中 NaN 置 0（仅占位，模型用 mask token 替代）。"""
    x = r.to_numpy(np.float32)
    obs = ~np.isnan(x)
    return torch.from_numpy(np.nan_to_num(x, nan=0.0)), torch.from_numpy(obs)
