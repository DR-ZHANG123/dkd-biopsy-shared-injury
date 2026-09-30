"""Stage 15（RRG-ID）的数据层：单元、比较层级（stratum）、特征缓存、fold 定义、共享轴协变量。

设计依据（notes/10_research_idea.md、results/13_celltype）：
- 疾病 vs 对照由一条共享轴主导；病种特异信号是叠加其上的小残差。
- 标签与队列强耦合（NEPTUNE 无 DKD；DKD–对照队列无其他 CKD）：一切监督比较只在同一 stratum
  （= 单元 × series；ERCB_GLOM_H1 内 LN / IgAN+HT / 其余病种分属三个杂交 series）内构造。
- 共享轴协变量 a：lib.injury 同法（非 DKD 病种 vs 对照、多来源平均），每个 fold 只用允许的来源单元，
  测试单元（E2 为测试联盟全部单元）永不作来源。
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from lib.injury import full_ranks, injury_axis, injury_score
from lib.repro import ROOT, load_config

CFG = load_config()
M2 = CFG["model2"]
LAB = M2["labels"]
sys.path.insert(0, str(ROOT / "scripts" / "stages"))


# ---------------------------------------------------------------- 单元与标签
@lru_cache(maxsize=None)
def _stage11_units() -> tuple[pd.DataFrame, dict]:
    s, units = import_module("11_injury_axis").load_samples()
    return s, units


def role_of(diag: str) -> str:
    if diag in LAB["head"]:
        return "head"
    if diag in LAB["neg_only"]:
        return "neg"
    if diag in LAB["control"]:
        return "ctrl"
    return "excl"


@lru_cache(maxsize=None)
def unit_table(comp: str) -> pd.DataFrame:
    """区室 comp 的 model2 单元样本表（index = sample_uid）：unit stratum series diagnosis role dup_group。

    同一 dup_group 出现在多个单元时只保留在 config 列表中靠前的单元；外部测试队列断言不出现。
    """
    s, units = _stage11_units()
    rows, seen = [], set()
    for u in M2["units"][comp]:
        assert u not in M2["never_use"], u
        idx = [x for x in units[u] if s.at[x, "dup_group"] not in seen]
        seen |= set(s.loc[idx, "dup_group"])
        d = s.loc[idx]
        rows.append(pd.DataFrame({"unit": u, "series": d.series.values, "diagnosis": d.diagnosis.values,
                                  "dup_group": d.dup_group.values}, index=pd.Index(idx, name="sample_uid")))
    t = pd.concat(rows)
    t["role"] = t.diagnosis.map(role_of)
    t = t[t.role != "excl"].copy()
    t["stratum"] = t.unit + "|" + t.series
    t["compartment"] = comp
    bad = set(t.series) & set(M2["never_use"]) | set(t.unit) & set(M2["never_use"])
    assert not bad, f"外部测试队列混入：{bad}"
    return t


def universe() -> list[str]:
    u = pd.read_csv(ROOT / "results/04_ranks/gene_universe.tsv", sep="\t")
    return u.gene.astype(str).tolist()


def features() -> pd.DataFrame:
    """两个区室全部 model2 样本在基因宇宙上的样本内秩（NaN = 该平台未测到）；缓存到 parquet。"""
    s, _ = _stage11_units()
    uids = sorted(set(unit_table("GLOM").index) | set(unit_table("TUB").index))
    genes = universe()
    key = hashlib.md5(("\n".join(uids) + "#" + "\n".join(genes)).encode()).hexdigest()
    p = ROOT / M2["features_cache"]
    kp = p.with_suffix(".key")
    if p.exists() and kp.exists() and kp.read_text() == key:
        return pd.read_parquet(p)
    f = full_ranks(pd.Index(uids), s).reindex(columns=genes).astype("float32")
    f.index.name = "sample_uid"
    p.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(p)
    p.with_suffix(".key").write_text(key)
    return f


# ---------------------------------------------------------------- fold
@dataclass
class M2Fold:
    family: str
    comp: str
    test_unit: str
    part: int                       # E3 的折号；E1/E2 为 -1
    train: pd.DataFrame             # unit_table 子集
    test: pd.DataFrame
    axis_sources: list[str] = field(default_factory=list)
    disease: str | None = None      # E4：该 fold 只评估这一个病种
    axis_exclude: str | None = None # E4：共享轴估计时把该病种从来源中剔除（与 DKD 情形一致：轴不含目标病种）

    @property
    def key(self) -> str:
        k = f"{self.family}_{self.comp}_{self.test_unit}" + (f"_p{self.part}" if self.part >= 0 else "")
        return k + (f"_{self.disease}" if self.disease else "")


def _consortium(u: str) -> str | None:
    for k, v in M2["consortia"].items():
        if u in v:
            return k
    return None


def _drop_dups(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return train[~train.dup_group.isin(set(test.dup_group))]


def _sources(comp: str, banned: set[str]) -> list[str]:
    return [u for u in CFG["injury"]["compartments"][comp]["source"] if u not in banned]


def make_folds(family: str, comp: str, drop_test_dkd: bool = True, test_units: list[str] | None = None
               ) -> list[M2Fold]:
    """E1 = 留一单元；E2 = 留一联盟（测试单元属 ERCB/NEPTUNE 时去掉该联盟全部单元）；E3 = 单元内 k 折。

    drop_test_dkd：dev 阶段为 True —— DKD 样本不进入任何测试集（只可能出现在训练集）。
    """
    t = unit_table(comp)
    if family == "E4":
        return _e4_folds(comp, t)
    out = []
    units = test_units or [u for u in M2["units"][comp] if (t[t.unit == u].role.isin(["head", "neg"])).any()]
    for u in units:
        test_all = t[t.unit == u]
        if family == "E1":
            banned = {u}
        elif family == "E2":
            c = _consortium(u)
            if c is None:
                continue
            banned = set(M2["consortia"][c])
        elif family == "E3":
            banned = {u}
        else:
            raise ValueError(family)
        if family in ("E1", "E2"):
            tr = _drop_dups(t[~t.unit.isin(banned)], test_all)
            te = test_all[test_all.diagnosis != "DKD"] if drop_test_dkd else test_all
            out.append(M2Fold(family, comp, u, -1, tr, te, _sources(comp, banned)))
            continue
        k = M2["eval"]["e3_folds"]
        sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=CFG["seed"])
        for p, (a, b) in enumerate(sgk.split(test_all, test_all.diagnosis, test_all.dup_group)):
            te = test_all.iloc[b]
            tr = pd.concat([t[t.unit != u], test_all.iloc[a]])
            tr = _drop_dups(tr, te)
            te = te[te.diagnosis != "DKD"] if drop_test_dkd else te
            out.append(M2Fold(family, comp, u, p, tr, te, _sources(comp, banned)))
    return out


def _e4_folds(comp: str, t: pd.DataFrame) -> list[M2Fold]:
    """E4 耦合模拟：把非 DKD 病种 d 放进与 DKD 同构的数据结构里，专门检验「标签–队列耦合」下的泛化。

    DKD 的结构：阳性主要来自「DKD–对照」队列（无其他病人、自成批次），NEPTUNE 有多种其他 CKD 但无 DKD，
    另一个 ERCB 批次只有少量 DKD。模拟：测试单元 = ERCB 单元 u；训练 = 其余单元，但
    - 指定 NEPTUNE series S_d（config e4.d_cohort[comp]，含对照）只保留 d 与对照 → 「d–对照队列」；
    - 其余 NEPTUNE 单元删去 d → 「无 d 的多病种队列」；
    - 其余单元（另一 ERCB 批次、DKD–对照队列等）不变；共享轴来源中剔除 d（轴不含目标病种）。
    """
    e4 = M2["eval"]["e4"]
    out = []
    for u in [x for x in M2["units"][comp] if x.startswith("ERCB_")]:
        test_all = t[(t.unit == u) & (t.diagnosis != "DKD")]
        for d in e4["diseases"]:
            if (test_all.diagnosis == d).sum() < M2["eval"]["min_test_pos"]:
                continue
            tr = _drop_dups(t[t.unit != u], test_all)
            nep = tr.unit.isin(M2["consortia"]["NEPTUNE"])
            dco = tr.unit == e4["d_cohort"][comp]
            keep = ~nep | (dco & tr.diagnosis.isin([d] + LAB["control"])) | (nep & ~dco & (tr.diagnosis != d))
            tr = tr[keep]
            out.append(M2Fold("E4", comp, u, -1, tr, test_all, _sources(comp, {u}), d, d))
    return out


# ---------------------------------------------------------------- 共享轴协变量
@lru_cache(maxsize=None)
def _axis(comp: str, sources: tuple[str, ...], exclude: str | None = None) -> pd.Series:
    s, units = _stage11_units()
    src = [units[u] if exclude is None else units[u][s.loc[units[u], "diagnosis"].values != exclude]
           for u in sources]
    return injury_axis(src, s)


@lru_cache(maxsize=None)
def _unit_score(comp: str, sources: tuple[str, ...], unit: str, exclude: str | None = None) -> pd.Series:
    """与 stage 11 相同：在整个单元（stage 11 的单元成员）的 full_ranks 上打分。"""
    s, units = _stage11_units()
    R = full_ranks(units[unit], s).dropna(axis=1)
    return injury_score(R, _axis(comp, sources, exclude))


def covariate(fold: M2Fold) -> pd.Series:
    """fold 内全部训练 + 测试样本的共享轴分数 a（sample_uid 索引）。"""
    src = tuple(fold.axis_sources)
    parts = []
    for u in sorted(set(fold.train.unit) | set(fold.test.unit)):
        parts.append(_unit_score(fold.comp, src, u, fold.axis_exclude))
    a = pd.concat(parts)
    a = a[~a.index.duplicated()]
    idx = fold.train.index.append(fold.test.index)
    out = a.reindex(idx)
    assert out.notna().all(), f"{fold.key}: {out.isna().sum()} 个样本缺共享轴分数"
    return out.rename("a")


def standardize_within(v: pd.Series, groups: pd.Series) -> pd.Series:
    g = v.groupby(groups.reindex(v.index).values)
    return ((v - g.transform("mean")) / (g.transform("std").fillna(1.0) + 1e-6)).rename(v.name)


def impute(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from lib.evalkit import median_impute_apply, median_impute_fit
    med = median_impute_fit(Xtr)
    return median_impute_apply(Xtr, med), median_impute_apply(Xte, med), med


def resolve_method(name: str) -> dict:
    if name in M2["methods"]:
        cfg = dict(M2["defaults"])
        cfg.update(M2["methods"][name])
    else:                                   # 设计迭代变体：相对迭代期的默认配置 dev_defaults（保持可复现）
        cfg = dict(M2["dev_defaults"])
        cfg.update(M2["dev_variants"][name])
    return cfg
