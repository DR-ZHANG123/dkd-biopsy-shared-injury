"""Stage 03：统一诊断与区室标签。

诊断仅供评估任务使用；预训练（stage 07）不读取本表的 diagnosis 列。
规则见 lib/meta_rules.py（ERCB / NEPTUNE 用标题中的病种代码，其余用 GEO 诊断字段原文）。
GSE30528/GSE30529 用同项目旧版 CEL 重处理 meta 的 group 字段（已按 CEL 校正样本对应）。
同一 dup_group（stage 02）内若诊断不一致，写入 label_conflicts.tsv，由 stage 04 按代表样本处理。
产出：results/03_labels/sample_labels.tsv、label_counts.tsv、label_conflicts.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.meta_rules import biopsy_id, classify_compartment, classify_diagnosis  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
INTERIM = ROOT / CFG["paths"]["interim"]
OUT = ROOT / "results" / "03_labels"
CEL_GROUP = {"DKD": "DKD", "Control": "CONTROL"}


def extra_fields(gse: str, row: pd.Series) -> dict:
    """任务相关的附加字段：GSE142025 的 DKD 分期（T3）、GSE175759 作者标注的技术离群样本。"""
    out = {"subtype": "", "technical_outlier": False}
    for v in (str(x) for k, x in row.items() if k.startswith("characteristics") and isinstance(x, str)):
        key, _, val = v.partition(":")
        if gse == "GSE142025" and key.strip() == "group":
            out["subtype"] = val.strip()
        if gse == "GSE175759" and key.strip() == "technical outlier":
            out["technical_outlier"] = val.strip().lower().startswith("outlier")
    return out


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(ROOT / "metadata" / "data_sources.tsv", sep="\t").set_index("accession")["role"]
    rows = []
    for p in sorted(INTERIM.glob("*_meta.tsv")):
        key = p.name.replace("_meta.tsv", "")
        gse = key.split("-")[0]
        m = pd.read_csv(p, sep="\t", dtype=str)
        for _, r in m.iterrows():
            comp = classify_compartment(r)
            bid = biopsy_id(gse, r, comp)
            if key.endswith("GPL571CEL"):
                diag, why = CEL_GROUP.get(r["group"], "UNKNOWN"), f"CEL meta group: {r['group']}"
            else:
                diag, why = classify_diagnosis(r, bid)
            rows.append({"sample_uid": f"{key}|{r['gsm']}", "series": key, "gse": gse, "gsm": r["gsm"],
                         "role": roles.get(gse, "unknown"), "compartment": comp, "diagnosis": diag,
                         "diag_source": why, "biopsy_id": bid, **extra_fields(gse, r),
                         "title": str(r.get("title", ""))[:100]})
    lab = pd.DataFrame(rows)
    unk = lab[(lab.diagnosis == "UNKNOWN") | (lab.compartment == "UNKNOWN")]
    lab.to_csv(OUT / "sample_labels.tsv", sep="\t", index=False)
    cnt = lab.groupby(["series", "role", "compartment", "diagnosis"]).size().rename("n").reset_index()
    cnt.to_csv(OUT / "label_counts.tsv", sep="\t", index=False)

    # 同一标本（dup_group）在不同 series 中的诊断是否一致
    st = pd.read_csv(ROOT / "results" / "02_overlap" / "sample_table.tsv", sep="\t", usecols=["sample_uid", "dup_group"])
    lab = lab.merge(st, on="sample_uid", how="left")
    g = lab.groupby("dup_group").diagnosis.agg(lambda s: "|".join(sorted(set(s))))
    bad = g[g.str.contains(r"\|")]
    conf = lab[lab.dup_group.isin(bad.index)].sort_values(["dup_group", "series"])
    conf[["dup_group", "sample_uid", "biopsy_id", "diagnosis", "diag_source", "title"]].to_csv(
        OUT / "label_conflicts.tsv", sep="\t", index=False)
    write_provenance("03_labels", sorted(INTERIM.glob("*_meta.tsv")) + [ROOT / "metadata" / "data_sources.tsv"],
                     [OUT / "sample_labels.tsv", OUT / "label_counts.tsv", OUT / "label_conflicts.tsv"], CFG["seed"],
                     {"n_samples": int(len(lab)), "n_unknown": int(len(unk)),
                      "n_dup_groups_with_label_conflict": int(len(bad)),
                      "conflict_patterns": bad.value_counts().to_dict()})
    print(cnt.pivot_table(index="series", columns="diagnosis", values="n", aggfunc="sum", fill_value=0).to_string())
    print("UNKNOWN rows:", len(unk))
    print("dup groups with conflicting diagnosis:", bad.value_counts().to_dict())


if __name__ == "__main__":
    main()
