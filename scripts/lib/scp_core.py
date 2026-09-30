"""Stage 19：SCP 核心基因的定义与读取（所有 19_* 脚本共用同一口径）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.repro import ROOT, load_config
from lib.scp import bh, re_meta, unit_effect

CFG = load_config()
SP = CFG["shared_program"]
OUT = ROOT / "results" / "19_shared_program"


def source_effects(comp: str, s: pd.DataFrame, units: dict, exclude: tuple = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """来源单元的 非 DKD 病人 vs 对照 g / var（基因须在全部来源中测到）。"""
    G, V = {}, {}
    for u in SP["sources"][comp]:
        if u in exclude:
            continue
        assert u not in SP["never_source"], u
        e = unit_effect(units[u], s)
        if e is None or e.n_pos.iloc[0] < SP["min_group"] or e.n_ctrl.iloc[0] < SP["min_group"]:
            continue
        G[u], V[u] = e.g, e.v
    G, V = pd.concat(G, axis=1), pd.concat(V, axis=1)
    keep = G.notna().all(1)
    return G[keep], V[keep]


def core_table(G: pd.DataFrame, V: pd.DataFrame) -> pd.DataFrame:
    """全部基因的 RE 合并 + 留一来源稳定性 + 核心标记。"""
    c = SP["core"]
    M = re_meta(G, V)
    M["fdr"] = bh(M.p.to_numpy())
    loo_ok, loo_g = [], {}
    for u in G.columns:
        m = re_meta(G.drop(columns=u), V.drop(columns=u))
        loo_g[u] = m.g_re
        loo_ok.append((np.sign(m.g_re) == np.sign(M.g_re)) & (m.p < c["loo_max_p"]) & (m.g_re.abs() >= c["loo_min_abs_g"]))
    L = pd.concat(loo_ok, axis=1)
    M["loo_pass_frac"] = L.mean(1)
    Lg = pd.DataFrame(loo_g)
    M["loo_min_abs_g_same_sign"] = (Lg.mul(np.sign(M.g_re), axis=0)).min(1)
    M["is_core"] = ((M.fdr < c["fdr"]) & (M.g_re.abs() >= c["min_abs_g"]) & (M.sign_frac >= c["sign_frac"])
                    & (M.loo_pass_frac == 1.0))
    M["direction"] = np.where(M.is_core, np.where(M.g_re > 0, "up", "down"), "")
    for u in G.columns:
        M[f"g_{u}"] = G[u]
    return M.rename_axis("gene").reset_index()


def load_core(comp: str) -> tuple[list[str], list[str]]:
    t = pd.read_csv(OUT / "core" / "core_genes.tsv", sep="\t")
    t = t[t.compartment == comp]
    return t.gene[t.direction == "up"].tolist(), t.gene[t.direction == "down"].tolist()


def load_core_table() -> pd.DataFrame:
    return pd.read_csv(OUT / "core" / "core_genes.tsv", sep="\t")


def loo_sets(comp: str, u: str, s: pd.DataFrame, units: dict) -> tuple[list[str], list[str]]:
    """单元 u 的分数所用核心集：若 u 是来源，则在其余来源上重估并取与全核心同样数量的上 / 下调基因
    （按 z 排序，要求其余来源同号）；否则返回全核心。"""
    up, dn = load_core(comp)
    if u not in SP["sources"][comp]:
        return up, dn
    G, V = source_effects(comp, s, units, exclude=(u,))
    M = re_meta(G, V)
    M = M[M.sign_frac == 1.0]
    return M.z.nlargest(len(up)).index.tolist(), M.z.nsmallest(len(dn)).index.tolist()
