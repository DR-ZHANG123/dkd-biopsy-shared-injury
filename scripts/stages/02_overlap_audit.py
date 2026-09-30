"""Stage 02：全局样本查重（研究方标本编号 + 表达相关两条证据）。

表达证据：对每两个 series（含 series 自身），在共有基因上做样本内秩，
  raw    = 秩的 Pearson（即 Spearman）；
  resid  = 各 series 内逐基因去均值并去掉前 resid_pcs 个主成分后的残差相关（去掉病种、批次等
           共享结构，只剩样本特异偏离；同一 RNA 的不同 CDF 处理版本 ≈0.6–1，不同样本 < 0.4）。
表达判重需同时满足：同区室、resid 互为最佳匹配、resid ≥ resid_min（series 内部 ≥ resid_min_within）、
推断性别不矛盾。
raw ≥ corr_threshold（方案中的 0.98）只作记录：肾组织样本间 raw Spearman 普遍 0.95–0.99，不能单独判重。
编号证据：ERCB / NEPTUNE 标题编号、Illumina 芯片条码（lib/meta_rules.py）。
dup_group = 连通分量，边 = 表达判重 ∪ 未被表达证据否定的同编号样本对（保守，冲突逐条列出）。
产出：results/02_overlap/{sample_table,pair_evidence,duplicate_pairs,id_expr_agreement,
      evidence_conflicts,duplicate_summary}.tsv
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.meta_rules import biopsy_id, classify_compartment  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
OV = CFG["overlap"]
INTERIM = ROOT / CFG["paths"]["interim"]
OUT = ROOT / "results" / "02_overlap"
Y_GENES = ["RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "USP9Y", "ZFY"]


def load_all() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    data, rows = {}, []
    for p in sorted(INTERIM.glob("*_expr.parquet")):
        key = p.name.replace("_expr.parquet", "")
        data[key] = pd.read_parquet(p)
        meta = pd.read_csv(INTERIM / f"{key}_meta.tsv", sep="\t", dtype=str)
        gse = key.split("-")[0]
        for _, r in meta.iterrows():
            comp = classify_compartment(r)
            rows.append({"sample_uid": f"{key}|{r['gsm']}", "series": key, "gse": gse, "gsm": r["gsm"],
                         "compartment": comp, "biopsy_id": biopsy_id(gse, r, comp),
                         "title": str(r.get("title", ""))[:80]})
    return data, pd.DataFrame(rows).set_index("sample_uid")


def infer_sex(expr: pd.DataFrame) -> pd.DataFrame:
    """样本内百分位秩：score = XIST − mean(Y 连锁基因)；|score| < sex_margin 不判定。"""
    r = expr.rank(pct=True)
    ys = [g for g in Y_GENES if g in r.index]
    if "XIST" not in r.index or len(ys) < 2:
        return pd.DataFrame({"sex_score": np.nan, "sex": "NA"}, index=expr.columns)
    score = r.loc["XIST"] - r.loc[ys].mean()
    m = OV["sex_margin"]
    sex = np.where(score > m, "F", np.where(score < -m, "M", "NA"))
    return pd.DataFrame({"sex_score": score.values, "sex": sex}, index=expr.columns)


def rank_z(expr: pd.DataFrame, genes: pd.Index, n_pcs: int) -> np.ndarray:
    """gene × sample 的样本内秩 → series 内逐基因去均值 → 去掉前 n_pcs 个主成分 → 按样本 z 标准化。
    n_pcs = 0 时即普通 Spearman 的输入。去主成分是为了去掉病种、批次等 series 内共享结构，
    只保留样本特异的偏离；同一 RNA 的不同处理版本在残差上仍高度相关，不同样本接近 0。
    该平台未测到的 NaN 基因用样本内中位秩填补。"""
    x = expr.loc[genes].rank(axis=0).values
    x = np.where(np.isnan(x), np.nanmedian(x, axis=0, keepdims=True), x)
    if n_pcs > 0:
        x = x - x.mean(axis=1, keepdims=True)
        k = min(n_pcs, x.shape[1] // 4)
        if k > 0:
            _, vec = np.linalg.eigh(x.T @ x)            # 样本空间 Gram 矩阵，等价于截断 SVD
            v = vec[:, -k:]
            x = x - (x @ v) @ v.T
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def compare(a: str, ea: pd.DataFrame, b: str, eb: pd.DataFrame, id_pairs: list[tuple[int, int]]
            ) -> tuple[list[dict], list[dict]]:
    """返回 (残差相关互为最佳匹配的样本对, 编号相同样本对的表达统计)。a == b 时比较 series 内部。"""
    genes = ea.index.intersection(eb.index)
    if len(genes) < OV["min_common_genes"]:
        return [], []
    raw = rank_z(ea, genes, 0).T @ rank_z(eb, genes, 0) / len(genes)
    res = rank_z(ea, genes, OV["resid_pcs"]).T @ rank_z(eb, genes, OV["resid_pcs"]) / len(genes)
    if a == b:
        np.fill_diagonal(raw, -1)
        np.fill_diagonal(res, -1)
    best_b, best_a = res.argmax(1), res.argmax(0)
    second_row = np.sort(res, axis=1)[:, -2] if res.shape[1] > 1 else np.full(len(res), np.nan)

    def rec(i: int, j: int) -> dict:
        mutual = bool(best_b[i] == j and best_a[j] == i)
        return {"series_a": a, "gsm_a": ea.columns[i], "series_b": b, "gsm_b": eb.columns[j],
                "n_genes": len(genes), "spearman": float(raw[i, j]), "resid_r": float(res[i, j]),
                "gap_to_second": float(res[i, j] - second_row[i]) if mutual else np.nan,
                "mutual_best": mutual, "resid_rank_in_row": int((res[i] > res[i, j]).sum()),
                "best_partner_b": eb.columns[best_b[i]], "best_partner_r": float(res[i, best_b[i]])}

    mutual = [rec(i, j) for i, j in enumerate(best_b) if best_a[j] == i and not (a == b and i > j)]
    return mutual, [rec(i, j) for i, j in id_pairs]


def id_index_pairs(samples: pd.DataFrame, a: str, ea: pd.DataFrame, b: str, eb: pd.DataFrame
                   ) -> list[tuple[int, int]]:
    ida = samples.loc[[f"{a}|{g}" for g in ea.columns], "biopsy_id"].values
    idb = samples.loc[[f"{b}|{g}" for g in eb.columns], "biopsy_id"].values
    pos_b: dict[str, list[int]] = {}
    for j, x in enumerate(idb):
        if isinstance(x, str):
            pos_b.setdefault(x, []).append(j)
    return [(i, j) for i, x in enumerate(ida) if isinstance(x, str) for j in pos_b.get(x, [])
            if not (a == b and i >= j)]


def call_duplicates(df: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    ua = df.series_a + "|" + df.gsm_a
    ub = df.series_b + "|" + df.gsm_b
    for s, u in (("a", ua), ("b", ub)):
        df[f"comp_{s}"] = samples.loc[u, "compartment"].values
        df[f"sex_{s}"] = samples.loc[u, "sex"].values
        df[f"id_{s}"] = samples.loc[u, "biopsy_id"].values
    df["same_compartment"] = df.comp_a == df.comp_b
    df["sex_ok"] = (df.sex_a == "NA") | (df.sex_b == "NA") | (df.sex_a == df.sex_b)
    df["pass_raw"] = df.spearman >= OV["corr_threshold"]
    # series 内部：残差共享同一组主成分，零分布更宽（实测不同患者可达 0.57），只接受近乎相同的技术重复
    thr = np.where(df.series_a == df.series_b, OV["resid_min_within"], OV["resid_min"])
    df["expr_dup"] = df.mutual_best & df.same_compartment & df.sex_ok & (df.resid_r >= thr)
    df["both_id"] = df.id_a.notna() & df.id_b.notna()
    df["same_id"] = df.both_id & (df.id_a == df.id_b)
    return df


def id_status(ev: pd.DataFrame) -> pd.DataFrame:
    """编号相同的样本对：confirmed（表达判重）/ contradicted（任一方在对方 series 中另有表达判定的
    同一标本）/ unconfirmed（表达无定论：同一标本在 U133A 与 Plus 2.0 上分别杂交时残差相关只有
    ≈0.40–0.45，series 内部的重复杂交也无法与 series 内的批次相似区分）。"""
    ua = ev.series_a + "|" + ev.gsm_a
    ub = ev.series_b + "|" + ev.gsm_b
    dup = ev[ev.expr_dup]
    partner = {}
    for x, sb, y, sa in zip(dup.series_a + "|" + dup.gsm_a, dup.series_b, dup.series_b + "|" + dup.gsm_b,
                            dup.series_a):
        partner[(x, sb)] = y
        partner[(y, sa)] = x
    contra = [(partner.get((x, sb), y) != y) or (partner.get((y, sa), x) != x)
              for x, sb, y, sa in zip(ua, ev.series_b, ub, ev.series_a)]
    ev["id_status"] = np.where(~ev.same_id, "", np.where(ev.expr_dup, "confirmed",
                               np.where(contra, "contradicted", "unconfirmed")))
    ns = lambda s: s.str.split(":").str[0]  # noqa: E731
    ev["comparable"] = ev.both_id & ev.same_compartment & (ns(ev.id_a.fillna("")) == ns(ev.id_b.fillna("")))
    return ev


def agreement(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """两条证据的一致率（只统计双方都有同一命名空间编号、同区室的样本对）。"""
    cmp_ = ev[ev.comparable]
    idp, exp_ = cmp_[cmp_.same_id], cmp_[cmp_.expr_dup]
    rows = [{"metric": "ID-matched pairs confirmed by expression", "n": len(idp),
             "n_agree": int((idp.id_status == "confirmed").sum())},
            {"metric": "ID-matched pairs contradicted by expression (different partner)", "n": len(idp),
             "n_agree": int((idp.id_status == "contradicted").sum())},
            {"metric": "ID-matched pairs without expression call (unconfirmed)", "n": len(idp),
             "n_agree": int((idp.id_status == "unconfirmed").sum())},
            {"metric": "expression-called pairs carrying identical IDs", "n": len(exp_),
             "n_agree": int(exp_.same_id.sum())},
            {"metric": "expression-called pairs where >=1 sample has no ID", "n": int((ev.expr_dup & ~ev.both_id).sum()),
             "n_agree": np.nan}]
    agree = pd.DataFrame(rows)
    agree["rate"] = agree.n_agree / agree.n.replace(0, np.nan)
    by_pair = cmp_.groupby(["series_a", "series_b"]).agg(
        n_id_pairs=("same_id", "sum"),
        n_confirmed=("id_status", lambda s: int((s == "confirmed").sum())),
        n_contradicted=("id_status", lambda s: int((s == "contradicted").sum())),
        n_unconfirmed=("id_status", lambda s: int((s == "unconfirmed").sum())),
        n_expr_dup=("expr_dup", "sum")).reset_index()
    by_pair = by_pair[(by_pair.n_id_pairs > 0) | (by_pair.n_expr_dup > 0)]
    return agree, by_pair


def dup_groups(samples: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    uid = pd.Index(samples.index)
    i = uid.get_indexer(edges.series_a + "|" + edges.gsm_a)
    j = uid.get_indexer(edges.series_b + "|" + edges.gsm_b)
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(uid), len(uid)))
    _, lab = connected_components(g, directed=False)
    # 组号按首次出现顺序重编，保证可复现
    order = pd.Series(lab).drop_duplicates()
    remap = {old: new for new, old in enumerate(order)}
    return pd.Series([f"D{remap[x]:05d}" for x in lab], index=uid)


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    data, samples = load_all()
    sex = pd.concat([infer_sex(v).set_axis([f"{k}|{c}" for c in v.columns]) for k, v in data.items()])
    samples = samples.join(sex)

    mutual, idrows = [], []
    for a, b in itertools.combinations_with_replacement(list(data), 2):
        m, r = compare(a, data[a], b, data[b], id_index_pairs(samples, a, data[a], b, data[b]))
        mutual += m
        idrows += r
    ev = pd.concat([pd.DataFrame(mutual), pd.DataFrame(idrows)], ignore_index=True)
    ev = ev.drop_duplicates(["series_a", "gsm_a", "series_b", "gsm_b"])
    ev = call_duplicates(ev, samples)

    ev = id_status(ev)
    ev.to_csv(OUT / "pair_evidence.tsv", sep="\t", index=False)
    agree, by_pair = agreement(ev)
    agree.to_csv(OUT / "id_expr_agreement.tsv", sep="\t", index=False)
    by_pair.to_csv(OUT / "id_expr_agreement_by_series_pair.tsv", sep="\t", index=False)
    conflicts = ev[ev.comparable & (ev.same_id != ev.expr_dup)]
    conflicts.to_csv(OUT / "evidence_conflicts.tsv", sep="\t", index=False)

    edges = ev[ev.expr_dup | (ev.same_id & ev.same_compartment & (ev.id_status != "contradicted"))]
    edges.to_csv(OUT / "duplicate_pairs.tsv", sep="\t", index=False)
    samples["dup_group"] = dup_groups(samples, edges)
    samples["n_in_group"] = samples.groupby("dup_group").dup_group.transform("size")
    samples.reset_index().to_csv(OUT / "sample_table.tsv", sep="\t", index=False)
    summ = edges.groupby(["series_a", "series_b"]).agg(
        n_pairs=("expr_dup", "size"), n_expr=("expr_dup", "sum"), n_id=("same_id", "sum")).reset_index()
    summ.to_csv(OUT / "duplicate_summary.tsv", sep="\t", index=False)
    outs = [OUT / f for f in ("sample_table.tsv", "pair_evidence.tsv", "duplicate_pairs.tsv",
                              "id_expr_agreement.tsv", "id_expr_agreement_by_series_pair.tsv",
                              "evidence_conflicts.tsv", "duplicate_summary.tsv")]
    write_provenance("02_overlap", sorted(INTERIM.glob("*_expr.parquet")) + sorted(INTERIM.glob("*_meta.tsv")),
                     outs, CFG["seed"],
                     {"n_samples": int(len(samples)), "n_unique_specimens": int(samples.dup_group.nunique()),
                      "n_duplicate_edges": int(len(edges)), "agreement": agree.to_dict("records")})
    print(summ.to_string())
    print(agree.to_string())
    print(by_pair.to_string())
    print("samples", len(samples), "unique specimens", samples.dup_group.nunique())


if __name__ == "__main__":
    main()
