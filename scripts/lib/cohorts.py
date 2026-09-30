"""评估 cohort 与 fold 的构建（stage 04 使用）。

ERCB：同一批 CEL 以不同 CDF 在多个 series 中重复存放 → 按 dup_group 去重，一个标本只出现一次，
再按芯片批次（HG-U133A = H1–H5 批；U133 Plus 2.0 = H7 批）分 cohort。
其余评估队列：一个 series 即一个 cohort。
"""
from __future__ import annotations

import pandas as pd


def ercb_cohorts(s: pd.DataFrame, ccfg: dict, n_genes: dict[str, int]) -> pd.DataFrame:
    """在每个「区室 × 芯片批次」内，series 版本排序：所在 CDF 版本（GPL）覆盖的唯一标本数 → 该 series
    覆盖的唯一标本数 → 基因数 → series 名。每个 dup_group 取排序最前的版本中的样本作代表
    （同版本内多个样本时取 GSM 最小者）。按 CDF 版本而非单个 series 排序，使同一 cohort 的
    病例与对照尽量来自同一 CDF（不同 CDF 版本的探针组成不同，会造成与诊断混杂的系统差异）。
    跨批次的 dup_group（同一标本在 U133A 与 Plus 2.0 上都杂交过）归入其样本数较多的批次。"""
    batch_of = ccfg["ercb_platform_batch"]
    e = s[s.gse.isin(ccfg["ercb_series"]) & s.compartment.isin(["GLOM", "TUB"])].copy()
    e["gpl"] = e.series.str.split("-").str[1]
    e["batch"] = e.gpl.map(batch_of)
    assert e.batch.notna().all(), f"ERCB 平台缺批次映射：{e.loc[e.batch.isna(), 'gpl'].unique()}"
    # 每个 dup_group 只归入一个批次：成员最多的批次（并列取批次名排序靠前者）
    nb = e.groupby(["dup_group", "batch"]).size().rename("n").reset_index()
    nb = nb.sort_values(["dup_group", "n", "batch"], ascending=[True, False, True]).drop_duplicates("dup_group")
    e = e[e.batch.values == e.dup_group.map(nb.set_index("dup_group").batch).values]
    out = []
    for (comp, batch), ec in e.groupby(["compartment", "batch"]):
        # 一个 GPL = 一个 CDF 版本；先按 CDF 版本的覆盖排序，再按 series 覆盖排序
        gcover = ec.groupby("gpl").dup_group.nunique()
        scover = ec.groupby("series").dup_group.nunique()
        order = sorted(scover.index, key=lambda k: (-gcover[k.split("-")[1]], -scover[k], -n_genes[k], k))
        prio = {k: i for i, k in enumerate(order)}
        ec = ec.assign(prio=ec.series.map(prio)).sort_values(["dup_group", "prio", "gsm"])
        rep = ec.drop_duplicates("dup_group")
        out.append(pd.DataFrame({"sample_uid": rep.index, "cohort": "ERCB_" + comp + "_" + rep.batch.values,
                                 "version_priority": rep.prio.values}))
    return pd.concat(out, ignore_index=True).set_index("sample_uid")


def assign_cohorts(s: pd.DataFrame, ccfg: dict, n_genes: dict[str, int]) -> pd.Series:
    """返回 sample_uid → cohort（只含评估 / 外部测试 cohort 的代表样本）。"""
    coh = ercb_cohorts(s, ccfg, n_genes).cohort
    single = {**{k: k for k in ccfg["single_series_cohorts"]}, **{k: k for k in ccfg["external"]}}
    extra = s[s.series.isin(single)]
    extra = extra.sort_values(["dup_group", "gsm"]).drop_duplicates("dup_group")
    return pd.concat([coh, extra.series.map(single)])


def eligible(s: pd.DataFrame, task: str, ccfg: dict) -> pd.Series:
    """T1：DKD vs CONTROL；T2：DKD vs 其他 CKD（config t2_negatives）。排除作者标注的技术离群样本。"""
    ok = ~s.technical_outlier.astype(bool)
    if task == "T1":
        return ok & s.diagnosis.isin(["DKD", "CONTROL"])
    return ok & s.diagnosis.isin(["DKD", *ccfg["t2_negatives"]])


def cohort_counts(s: pd.DataFrame, cohort: pd.Series, ccfg: dict) -> pd.DataFrame:
    rows = []
    for c, idx in cohort.groupby(cohort).groups.items():
        x = s.loc[idx]
        other = x.diagnosis.isin(ccfg["t2_negatives"])
        rows.append({
            "cohort": c, "compartment": x.compartment.mode().iat[0],
            "technology": "|".join(sorted(x.technology.unique())),
            "role": "external_test" if c in ccfg["external"] else "evaluation",
            "n_specimens": len(x), "n_DKD": int((x.diagnosis == "DKD").sum()),
            "n_CONTROL": int((x.diagnosis == "CONTROL").sum()), "n_other_CKD": int(other.sum()),
            "n_excluded_dx": int((~x.diagnosis.isin(["DKD", "CONTROL", *ccfg["t2_negatives"]])).sum()),
            "n_technical_outlier": int(x.technical_outlier.astype(bool).sum()),
            "other_CKD_breakdown": ";".join(f"{k}={v}" for k, v in x.diagnosis[other].value_counts().items()),
            "excluded_dx_breakdown": ";".join(f"{k}={v}" for k, v in x.diagnosis[
                ~x.diagnosis.isin(["DKD", "CONTROL", *ccfg["t2_negatives"]])].value_counts().items()),
            "series_versions": ";".join(f"{k}={v}" for k, v in x.series.value_counts().items()),
        })
    t = pd.DataFrame(rows)
    n = ccfg["min_class_n"]
    el1 = {c: eligible(s.loc[i], "T1", ccfg) for c, i in cohort.groupby(cohort).groups.items()}
    el2 = {c: eligible(s.loc[i], "T2", ccfg) for c, i in cohort.groupby(cohort).groups.items()}
    t["T1_n_DKD"] = [int((s.loc[el1[c][el1[c]].index, "diagnosis"] == "DKD").sum()) for c in t.cohort]
    t["T1_n_CONTROL"] = [int((s.loc[el1[c][el1[c]].index, "diagnosis"] == "CONTROL").sum()) for c in t.cohort]
    t["T2_n_DKD"] = [int((s.loc[el2[c][el2[c]].index, "diagnosis"] == "DKD").sum()) for c in t.cohort]
    t["T2_n_other"] = [int((s.loc[el2[c][el2[c]].index, "diagnosis"] != "DKD").sum()) for c in t.cohort]
    t["T1_adequate"] = (t.T1_n_DKD >= n) & (t.T1_n_CONTROL >= n)
    t["T2_adequate"] = (t.T2_n_DKD >= n) & (t.T2_n_other >= n)
    # 外部测试 cohort 两类都 ≥ external_min_class_n 时仍建 fold，但只作描述性报告（不进入合并估计）
    m = ccfg["external_min_class_n"]
    ext = t.role == "external_test"
    t["T1_descriptive_only"] = ext & ~t.T1_adequate & (t.T1_n_DKD >= m) & (t.T1_n_CONTROL >= m)
    return t.sort_values(["compartment", "role", "cohort"]).reset_index(drop=True)


def make_folds(s: pd.DataFrame, cohort: pd.Series, table: pd.DataFrame, ccfg: dict) -> pd.DataFrame:
    """LOCO：每个合格评估 cohort 轮流作 test，其余同区室合格 cohort 作 train；外部 cohort 只作 test
    （train = 该区室全部合格评估 cohort）。train 中与 test 同 dup_group 的样本剔除。"""
    rows = []
    fold_comp = ccfg["external"]
    for task in ("T1", "T2"):
        ok = table[table[f"{task}_adequate"]]
        for comp in ("GLOM", "TUB"):
            ev = ok[(ok.compartment == comp) & (ok.role == "evaluation")].cohort.tolist()
            ext_t = table[(table.role == "external_test") & (table.T1_adequate | table.T1_descriptive_only)]
            ext = [c for c in ext_t.cohort if fold_comp[c] == comp] if task == "T1" else []
            tests = [(c, c, [x for x in ev if x != c]) for c in ev] if len(ev) >= 2 else []
            tests += [(f"EXT_{c}", c, ev) for c in ext if ev]
            for fold, test_c, train_cs in tests:
                te = _members(s, cohort, [test_c], task, ccfg)
                tr = _members(s, cohort, train_cs, task, ccfg)
                tr = tr[~s.loc[tr, "dup_group"].isin(set(s.loc[te, "dup_group"]))]
                for split, idx in (("train", tr), ("test", te)):
                    rows += [{"task": task, "compartment": comp, "fold": fold, "test_cohort": test_c,
                              "sample_uid": u, "split": split} for u in idx]
    return pd.DataFrame(rows)


def _members(s: pd.DataFrame, cohort: pd.Series, cs: list[str], task: str, ccfg: dict) -> pd.Index:
    idx = cohort.index[cohort.isin(cs)]
    el = eligible(s.loc[idx], task, ccfg)
    return idx[el.values]
