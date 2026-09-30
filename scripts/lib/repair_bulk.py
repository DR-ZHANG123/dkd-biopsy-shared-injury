"""Stage 20 bulk 工具：BayesPrism 状态层份额与状态程序秩打分。

- theta_shares()：θ_state（stage 16 run/<tag>，指定参照）→ 每样本 修复失败 / 各状态 占 PT、TAL 类型的份额与占组织的绝对 θ。
- program_sets()：stage 20a 打分用程序（snRNA 定义）；variant = full / noSCP（剔除全部 SCP 核心基因，避免与 SCP 分数的基因重叠造成的机械相关）。
- program_scores()：样本内百分位秩上 上调 − 下调 均值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.repair_state import OUT, RS, repair_label
from lib.repro import ROOT
from lib.scp_core import load_core_table

SCORED = {"PT": ["aPT", "frPT", "rfPT", "dPT", "cycPT"], "TAL": ["aTAL", "frTAL", "rfTAL", "dTAL"]}


def theta_shares(sample_uids: pd.Index) -> pd.DataFrame:
    B = RS["bulk"]
    t = pd.read_csv(ROOT / "results" / "16_deconv" / "run" / B["theta_run"] / "theta_state.tsv", sep="\t")
    t = t[(t.reference == B["reference"]) & t.sample_uid.isin(sample_uids)]
    if t.empty:                                             # 该队列没有做 BayesPrism（KPMP 切片 bulk、GSE166239）
        return pd.DataFrame(index=pd.Index([], name="sample_uid"))
    W = t.pivot_table(index="sample_uid", columns="state", values="theta", aggfunc="sum").fillna(0.0)
    out = pd.DataFrame(index=W.index)
    for lin, d in RS["lineages"].items():
        cols = {st: [f for f in fs if f in W.columns] for st, fs in d.items()}
        tot = sum(W[c].sum(1) for c in cols.values() if c)
        out[f"theta_{lin}"] = tot
        per = {st: W[c].sum(1) for st, c in cols.items() if c}
        per[repair_label(lin)] = sum(per[s] for s in RS["repair_states"][lin] if s in per)
        per["altered"] = sum(v for s, v in per.items() if s not in ("normal", repair_label(lin)))
        for st, v in per.items():
            if st == "normal":
                continue
            out[f"share_{lin}:{st}"] = v / tot.where(tot > 0)
            out[f"abs_{lin}:{st}"] = v
    return out


def program_sets(variant: str) -> dict[str, tuple[list[str], list[str]]]:
    z = pd.read_csv(OUT / "programs" / "program_sets.tsv", sep="\t")
    excl = set(load_core_table().gene) if variant == "noSCP" else set()
    out = {}
    for lin, sts in SCORED.items():
        for st in sts:
            g = z[(z.lineage == lin) & (z.state == st) & ~z.gene.isin(excl)]
            if len(g):
                out[f"{lin}:{st}"] = (g.gene[g.direction == "up"].tolist(), g.gene[g.direction == "down"].tolist())
    return out


def program_scores(R: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for variant in ("full", "noSCP"):
        for k, (up, dn) in program_sets(variant).items():
            u = [x for x in up if x in R.columns]
            d = [x for x in dn if x in R.columns]
            if len(u) >= 10 and len(d) >= 10:
                cols[f"prog_{variant}_{k}"] = R[u].mean(1) - R[d].mean(1)
                cols[f"progup_{variant}_{k}"] = R[u].mean(1)
    return pd.DataFrame(cols, index=R.index)


def hedges_var(g: float, na: int, nb: int) -> float:
    return (na + nb) / (na * nb) + g ** 2 / (2 * (na + nb))
