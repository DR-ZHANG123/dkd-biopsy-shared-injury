"""GEO 文件解析与基因 ID 统一（stage 01 使用）。

所有平台先映射到 NCBI Entrez GeneID，再用同一份 NCBI gene_info 转成当前官方 symbol，
避免不同年代平台注释里的旧 symbol 造成跨平台基因名不一致。
"""
from __future__ import annotations

import gzip
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


def read_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (probe × sample 表达矩阵, 样本注释表)。重复的 !Sample_ 键加 .1/.2 后缀。"""
    meta_rows: dict[str, list[str]] = {}
    table_lines: list[str] = []
    in_table = False
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                table_lines.append(line)
            elif line.startswith("!Sample_"):
                key, *vals = line.rstrip("\n").split("\t")
                vals = [v.strip('"') for v in vals]
                base = key[len("!Sample_"):]
                k, i = base, 1
                while k in meta_rows:
                    k = f"{base}.{i}"
                    i += 1
                meta_rows[k] = vals
    meta = pd.DataFrame(meta_rows)
    if len(table_lines) <= 1:          # RNA-seq series：矩阵为空，只有表头
        return pd.DataFrame(), meta
    expr = pd.read_csv(StringIO("".join(table_lines)), sep="\t", index_col=0, low_memory=False)
    expr.index = expr.index.astype(str).str.strip('"')
    expr.columns = [c.strip('"') for c in expr.columns]
    return expr.apply(pd.to_numeric, errors="coerce"), meta


def read_soft_platform_table(path: Path) -> tuple[pd.DataFrame, str]:
    """流式解析 GPL family SOFT 中 !platform_table_begin/end 之间的注释表（读到表尾即停）。"""
    lines: list[str] = []
    title = ""
    inside = False
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Platform_title"):
                title = line.split("=", 1)[1].strip()
            elif line.startswith("!platform_table_begin"):
                inside = True
            elif line.startswith("!platform_table_end"):
                break
            elif inside:
                lines.append(line)
    if not lines:
        raise ValueError(f"{path}: 未找到 platform_table")
    tab = pd.read_csv(StringIO("".join(lines)), sep="\t", dtype=str, low_memory=False)
    tab["ID"] = tab["ID"].astype(str).str.strip()
    return tab.set_index("ID"), title


class GeneIndex:
    """NCBI gene_info：Entrez → 当前 symbol；symbol/唯一别名 → Entrez。"""

    def __init__(self, gene_info: Path):
        gi = pd.read_csv(gene_info, sep="\t", usecols=["GeneID", "Symbol", "Synonyms"], dtype=str)
        self.e2s = dict(zip(gi.GeneID, gi.Symbol))
        self.s2e = dict(zip(gi.Symbol, gi.GeneID))
        syn = gi[gi.Synonyms != "-"].assign(s=gi.Synonyms.str.split("|")).explode("s")
        syn = syn[~syn.s.isin(self.s2e)]
        uniq = syn.groupby("s").GeneID.nunique()
        syn = syn[syn.s.isin(uniq[uniq == 1].index)]
        self.syn2e = dict(zip(syn.s, syn.GeneID))

    def symbol_to_entrez(self, sym: str | float) -> str | None:
        if not isinstance(sym, str) or not sym:
            return None
        return self.s2e.get(sym) or self.syn2e.get(sym)


def _single_entrez(val: str | float) -> str | None:
    """'1234' → '1234'；'1234 /// 5678'（多基因探针）或空 → None。"""
    if not isinstance(val, str):
        return None
    ids = {v.strip().split(".")[0] for v in re.split(r"///|//|,", val) if v.strip() not in ("", "---")}
    ids = {i for i in ids if i.isdigit()}
    return ids.pop() if len(ids) == 1 else None


def _entrez_from_gene_assignment(val: str | float) -> str | None:
    """Affymetrix HTA/ST gene_assignment：'NM_x // SYM // desc // band // ENTREZ /// ...'。"""
    if not isinstance(val, str) or val.strip() in ("", "---"):
        return None
    ids = set()
    for rec in val.split("///"):
        parts = [p.strip() for p in rec.split("//")]
        if len(parts) >= 5 and parts[4].isdigit():
            ids.add(parts[4])
    return ids.pop() if len(ids) == 1 else None


def platform_entrez(tab: pd.DataFrame, probes: pd.Index) -> tuple[pd.Series, str]:
    """probe → Entrez。返回 (映射, 使用的注释列)。多基因探针置 None。"""
    ids = pd.Index(probes.astype(str))
    if ids.str.fullmatch(r"\d+_at").mean() > 0.8:            # BrainArray ENTREZG 定制 CDF
        return pd.Series(np.where(ids.str.fullmatch(r"\d+_at"), ids.str[:-3], None), index=probes), "ID(_at)"
    for col in ("ENTREZ_GENE_ID", "Entrez_Gene_ID", "GeneID"):
        if col in tab.columns:
            ent = tab[col].map(_single_entrez)
            return pd.Series(ent.reindex(ids).values, index=probes), col
    if "ORF" in tab.columns and tab["ORF"].dropna().str.fullmatch(r"\d+").mean() > 0.8:
        ent = tab["ORF"].map(_single_entrez)                   # BrainArray v10：ORF 列即 Entrez
        return pd.Series(ent.reindex(ids).values, index=probes), "ORF"
    if "gene_assignment" in tab.columns:
        ent = tab["gene_assignment"].map(_entrez_from_gene_assignment)
        return pd.Series(ent.reindex(ids).values, index=probes), "gene_assignment"
    raise KeyError(f"找不到 Entrez 注释列：{list(tab.columns)}")


def collapse_to_symbol(expr: pd.DataFrame, entrez: pd.Series, gidx: GeneIndex) -> tuple[pd.DataFrame, dict]:
    """Entrez → 当前 symbol；同一基因多探针时保留跨样本方差最大的探针（与旧版一致）。"""
    sym = entrez.map(lambda e: gidx.e2s.get(e) if isinstance(e, str) else None)
    keep = sym.notna().values & expr.notna().any(axis=1).values
    e = expr.loc[keep].copy()
    s = sym[keep].values
    var = e.var(axis=1).values
    order = np.lexsort((-var, s))                               # 每个 symbol 内按方差降序
    e = e.iloc[order]
    s = s[order]
    first = ~pd.Series(s).duplicated().values
    out = e.iloc[first]
    out.index = pd.Index(s[first], name="gene")
    stats = {"n_probes": int(len(expr)), "n_probes_entrez": int(entrez.notna().sum()),
             "n_probes_symbol": int(keep.sum()), "n_genes": int(len(out))}
    return out.sort_index(), stats
