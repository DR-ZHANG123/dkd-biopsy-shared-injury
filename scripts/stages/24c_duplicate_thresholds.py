"""Stage 24c：重复标本判定阈值的敏感性。

在 stage 02 已保存的样本对证据（results/02_overlap/pair_evidence.tsv：各 series 对的互为最佳残差匹配 + 编号匹配对）上，
以 pathways_sensitivity.duplicates 网格内的跨 series / series 内残差相关阈值重新判定表达重复，其余规则与 stage 02 相同
（同区室、互为最佳匹配、性别不矛盾；编号匹配且未被表达矛盾的对也连边）。报告：表达判重对数、编号–表达一致率、
独立标本数、与主分析标本划分不同的样本数，以及分析队列内部 / 分析队列之间出现的新重复对（泄漏检查）。
产出 results/24_pathways_sensitivity/duplicates/threshold_grid.tsv
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
OUT = ROOT / "results" / "24_pathways_sensitivity" / "duplicates"
S02 = importlib.import_module("02_overlap_audit")


def recall(ev0: pd.DataFrame, samples: pd.DataFrame, rmin: float, rwithin: float) -> tuple[pd.DataFrame, pd.Series]:
    S02.OV["resid_min"], S02.OV["resid_min_within"] = rmin, rwithin
    keep = [c for c in ev0.columns if c not in {"comp_a", "sex_a", "id_a", "comp_b", "sex_b", "id_b", "same_compartment",
                                                  "sex_ok", "pass_raw", "expr_dup", "both_id", "same_id", "id_status",
                                                  "comparable"}]
    ev = S02.id_status(S02.call_duplicates(ev0[keep].copy(), samples))
    edges = ev[ev.expr_dup | (ev.same_id & ev.same_compartment & (ev.id_status != "contradicted"))]
    return ev, S02.dup_groups(samples, edges)


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    ev0 = pd.read_csv(ROOT / "results/02_overlap/pair_evidence.tsv", sep="\t", dtype={"id_a": str, "id_b": str})
    samples = pd.read_csv(ROOT / "results/02_overlap/sample_table.tsv", sep="\t").set_index("sample_uid")
    samples["sex"] = samples.sex.fillna("NA")
    main_grp = samples.dup_group
    coh = pd.read_csv(ROOT / "data/processed/cohorts.tsv", sep="\t").set_index("sample_uid").cohort
    grid = CFG["pathways_sensitivity"]["duplicates"]
    rows = []
    for rmin in grid["resid_min"]:
        for rw in grid["resid_min_within"]:
            ev, grp = recall(ev0, samples, rmin, rw)
            agree, _ = S02.agreement(ev)
            a = agree.set_index("metric").rate
            # 标本划分与主分析是否一致：两种划分下「同组」关系不同的样本
            pair = pd.DataFrame({"main": main_grp, "alt": grp})
            changed = pair.groupby("alt").main.transform("nunique").gt(1) | pair.groupby("main").alt.transform("nunique").gt(1)
            g = grp.reindex(coh.index)
            within = int(g.groupby(coh).apply(lambda s: s.duplicated().sum()).sum())
            across = int(g.drop_duplicates().size != g.groupby(coh).apply(lambda s: s.drop_duplicates()).size)
            n_across = int(g.groupby(coh).apply(lambda s: s.drop_duplicates()).reset_index(drop=True).duplicated().sum())
            rows.append({"resid_min": rmin, "resid_min_within": rw, "n_expr_pairs": int(ev.expr_dup.sum()),
                         "id_pairs_confirmed": a.get("ID-matched pairs confirmed by expression"),
                         "expr_pairs_with_identical_id": a.get("expression-called pairs carrying identical IDs"),
                         "n_unique_specimens": int(grp.nunique()), "n_samples_regrouped": int(changed.sum()),
                         "dup_pairs_within_cohorts": within, "specimens_shared_between_cohorts": n_across,
                         "any_shared_between_cohorts": bool(across)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "threshold_grid.tsv", sep="\t", index=False)
    write_provenance("24_pathways_sensitivity/duplicates",
                     [ROOT / "results/02_overlap/pair_evidence.tsv", ROOT / "results/02_overlap/sample_table.tsv",
                      ROOT / "data/processed/cohorts.tsv"], [OUT / "threshold_grid.tsv"], CFG["seed"])
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
