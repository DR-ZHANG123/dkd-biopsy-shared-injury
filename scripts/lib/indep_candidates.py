"""Stage 17a 用：独立测试集候选的读取（KPMP LMD / 活检切片 bulk、GEO 候选），统一成
gene（当前 NCBI symbol）× sample 的 log2 尺度矩阵 + 样本表。只读诊断字段用于统计样本构成，不做任何模型评估。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from lib.geo import GeneIndex, collapse_to_symbol, platform_entrez, read_series_matrix, read_soft_platform_table

# KPMP 列名中的区段代码 → 标准区段；GLOM/TI 与 stage 15 的 GLOM/TUB 对应，Bulk = 切片整体（WHOLE）
REGION_PATTERNS = [("GLOM_noBC", r"noBC|no[ -]B\.?C"), ("Bulk", r"\bBulk\b"), ("GLOM", r"\b(Glom|G)\b"),
                   ("TI", r"\bTi\b|\bTI\b"), ("PT", r"\b(S1S2|ProxTub|PT)\b"), ("TAL", r"\b(TAL|Tal)\b"),
                   ("DCT", r"\bDCT\b"), ("CD", r"\bCD\b"), ("INT", r"\b(INT|Int)\b"), ("Vessel", r"Vessel")]
REGION_COMP = {"GLOM": "GLOM", "TI": "TUB", "Bulk": "WHOLE"}
KPMP_CLIN = ["redcap_id", "enrollment_category", "primary_adjudicated_category", "sample_type", "age_binned", "sex",
             "race", "baseline_egfr", "proteinuria", "albuminuria", "a1c", "diabetes_history", "diabetes_duration",
             "hypertension_history", "hypertension_duration", "on_raas_blockade", "kdigo_stage"]


def region_of(col: str) -> str:
    c = col.replace("_", " ").replace("-", " ").replace(".", ". ")
    for k, pat in REGION_PATTERNS:
        if re.search(pat, c):
            return k
    return "NA"


def kpmp_diag(adj: str | float, enroll: str) -> str:
    """KPMP 裁定类别 → 本项目诊断标签（与 stage 03 的命名一致）。"""
    if enroll == "Healthy Reference":
        return "CONTROL"
    return {"Diabetic Kidney Disease": "DKD", "Hypertensive Kidney Disease": "HT", "Other": "OTHER",
            "Acute Tubular Injury": "AKI_ATI", "Acute Interstitial Nephritis": "AKI_AIN"}.get(adj, "UNDETERMINED")


def _read_counts(path: Path) -> pd.DataFrame:
    hdr = path.open(encoding="utf-8-sig").readline()
    sep = "\t" if "\t" in hdr else ","
    d = pd.read_csv(path, sep=sep, index_col=0, encoding="utf-8-sig", low_memory=False)
    d = d.drop(columns=[c for c in ("Chr", "Start", "End", "Strand", "Length", "Description") if c in d])
    d = d.loc[:, [c for c in d.columns if not re.match(r"^10_\d\d_S\d+$", c)]]   # HiSeq 文件中的 10_xx 列为 log 值，非计数
    d.columns = [c.strip() for c in d.columns]
    d = d.apply(pd.to_numeric, errors="coerce")
    # 只保留计数列：个别开放文件（34-10050 Bulk_S25）的值为 log 尺度（含负值、非整数），不是计数
    d = d.loc[:, [c for c in d.columns if (d[c] >= 0).all() and (d[c].dropna() % 1 == 0).mean() > 0.99]]
    d.index = d.index.astype(str)
    return d.groupby(level=0).sum(min_count=1)       # 个别文件同一 symbol 多行（不同转录本座位）：计数求和


def _log2cpm_symbols(counts: pd.DataFrame, entrez: pd.Series, gidx: GeneIndex) -> pd.DataFrame:
    """与 stage 01 的 RNA-seq 处理相同：文库大小 = 映射前全部行之和，log2(CPM+1)，同一 symbol 在计数尺度求和。"""
    sym = entrez.map(lambda e: gidx.e2s.get(e) if isinstance(e, str) else None)
    cpm = counts / counts.sum(axis=0) * 1e6
    cpm = cpm.loc[sym.notna().values]
    cpm.index = sym[sym.notna()].values
    return np.log2(cpm.groupby(level=0).sum() + 1).rename_axis("gene")


def load_kpmp(raw: Path, gidx: GeneIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    man = pd.concat([pd.read_csv(raw / f"manifest_{t}.tsv", sep="\t").assign(strategy=t) for t in ("regional", "bulk")])
    ck = pd.read_csv(raw / "checksums.tsv", sep="\t")
    meta = man.drop_duplicates("file_name").set_index("file_name")
    cols, rows, seen = [], [], {}
    for f in ck.file[ck.file.str.endswith(".csv")]:
        d = _read_counts(raw / "files" / f)
        m = meta.loc[f]
        for c in d.columns:
            v = d[c].fillna(0).to_numpy()
            h = hashlib.md5(v.tobytes()).hexdigest()
            if h in seen:                      # 同一文件在两个 strategy 下重复登记
                continue
            seen[h] = c
            uid = f"KPMP|{m.redcap_id}|{c}"
            cols.append(d[c].rename(uid))
            rows.append({"sample_uid": uid, "cohort": "KPMP", "participant": m.redcap_id, "column": c,
                         "region": region_of(c), "file": f, "strategy": m.strategy,
                         **{k: m.get(k) for k in KPMP_CLIN}})
    pilot = next((raw / "pilot").glob("Pilot Sub-segmental RNAseq raw counts*.csv"))
    d = _read_counts(pilot)
    for c in d.columns:
        pid = re.sub(r"^(Bulk|Glom|ProxTub|Tal|TAL|DCT|CD|INT|Int|TI|Ti)_", "", c)
        pid = re.search(r"(1[78]-\d{4}|IU1[78]\d{5}|IU1[78]-\d{4})", pid).group(1).replace("IU17-", "IU170").replace("IU18-", "IU180")
        pid = {"IU1700001": "17-0001", "IU1700002": "17-0002", "IU1800006": "18-0006"}.get(pid, pid)
        uid = f"KPMP|{pid}|{c}"
        cols.append(d[c].rename(uid))
        rows.append({"sample_uid": uid, "cohort": "KPMP", "participant": pid, "column": c,
                     "region": region_of(c.split("_")[0]), "file": pilot.name, "strategy": "regional_pilot",
                     "enrollment_category": "Healthy Reference", "sample_type": "Nephrectomy (pilot reference)"})
    counts = pd.concat(cols, axis=1)
    ent = pd.Series([gidx.symbol_to_entrez(s) for s in counts.index.astype(str)], index=counts.index)
    expr = _log2cpm_symbols(counts, ent, gidx)
    s = pd.DataFrame(rows).set_index("sample_uid")
    s["diagnosis"] = [kpmp_diag(a, e) for a, e in zip(s.primary_adjudicated_category, s.enrollment_category)]
    s["compartment"] = s.region.map(REGION_COMP).fillna("OTHER_SEGMENT")
    s["library_reads"] = counts.sum(axis=0).reindex(s.index).values
    return expr, s


def load_gse162830(raw: Path, gidx: GeneIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """作者上传的分位数标准化表（NCBI 统一 counts 缺 7 个样本）。分位数标准化不改变样本内秩，
    log2(x+1) 后与其他 RNA-seq 同尺度处理；行 = symbol。"""
    c = pd.read_csv(raw / "GSE162830_ING_quantile_normalized_final.csv.gz", index_col=0)
    c.index = c.index.astype(str)
    c = c.groupby(level=0).sum()
    _, m = read_series_matrix(raw / "GSE162830_series_matrix.txt.gz")
    dis = m.characteristics_ch1.str.replace("disease: ", "")
    lab = dis.map({"Diabetic Nephropathy": "DKD", "Reference": "CONTROL",
                   "Idiopathic Nodular Glomerulosclerosis": "ING"})
    pid = m.title.str.replace("Laser microdissection ", "")
    assert set(pid) == set(c.columns), "GSE162830：标准化表列名与 GEO 标题不一致"
    s = pd.DataFrame({"cohort": "GSE162830", "participant": pid.values, "column": m.geo_accession.values,
                      "region": "GLOM", "compartment": "GLOM", "diagnosis": lab.values,
                      "tissue": m["characteristics_ch1.1"].str.replace("tissue type: ", "").values},
                     index=pd.Index("GSE162830|" + m.geo_accession, name="sample_uid"))
    sym = pd.Series(c.index.map(lambda x: gidx.e2s.get(gidx.symbol_to_entrez(x) or "")), index=c.index)
    e = c[pid.values].loc[sym.notna().values]
    e.index = sym[sym.notna()].values
    e = np.log2(e.groupby(level=0).sum() + 1).rename_axis("gene")
    e.columns = s.index
    return e, s


def _ensembl_to_entrez(gene_info: Path) -> dict[str, str]:
    gi = pd.read_csv(gene_info, sep="\t", usecols=["GeneID", "dbXrefs"], dtype=str)
    out: dict[str, str] = {}
    for g, x in zip(gi.GeneID, gi.dbXrefs):
        for ens in re.findall(r"Ensembl:(ENSG\d+)", x or ""):
            out.setdefault(ens, g)
    return out


def load_gse166239(raw: Path, gidx: GeneIndex, gene_info: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = pd.read_csv(raw / "GSE166239_Nordbo_et_al_counts.txt.gz", sep="\t", index_col=0)
    _, m = read_series_matrix(raw / "GSE166239_series_matrix.txt.gz")
    ch = {k: m.filter(regex="^characteristics").apply(lambda r: next((v.split(": ", 1)[1] for v in r
                                                                      if str(v).startswith(k + ":")), None), axis=1)
          for k in ("diagnosis", "age", "gender", "egfr (ml/min/m2)", "proteinuria (g/d)", "diabetes", "hypertension")}
    lab = ch["diagnosis"].map(lambda x: "DKD" if "diabetic" in x else "HT" if "hypertensive" in x else "CONTROL")
    s = pd.DataFrame({"cohort": "GSE166239", "participant": m.title.values, "column": m.title.values,
                      "region": "Bulk", "compartment": "WHOLE", "diagnosis": lab.values,
                      **{k: v.values for k, v in ch.items() if k != "diagnosis"}},
                     index=pd.Index("GSE166239|" + m.geo_accession, name="sample_uid"))
    e2e = _ensembl_to_entrez(gene_info)
    ent = pd.Series([e2e.get(i.split(".")[0]) for i in c.index], index=c.index)
    e = _log2cpm_symbols(c[m.title], ent, gidx)
    e.columns = s.index
    s["library_reads"] = c[m.title].sum().values
    return e, s


def load_gse45980(raw: Path, gidx: GeneIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, m = read_series_matrix(raw / "GSE45980_series_matrix.txt.gz")
    tab, _ = read_soft_platform_table(raw / "GPL13497_family.soft.gz")
    ent, _ = platform_entrez(tab.rename(columns={"GENE": "GeneID"}), x.index)   # Agilent：GENE 列 = Entrez
    e, _ = collapse_to_symbol(x, ent, gidx)
    ch = m.filter(regex="^characteristics")
    get = lambda k: ch.apply(lambda r: next((v.split(": ", 1)[1] for v in r if str(v).startswith(k)), None), axis=1)  # noqa: E731
    dx = get("diagnosis")
    lab = dx.map(lambda d: "DKD" if d and "Diabetic" in d else "HT" if d and "Hypertensive" in d else "OTHER_CKD")
    s = pd.DataFrame({"cohort": "GSE45980", "participant": m.title.values, "column": m.geo_accession.values,
                      "region": "Bulk", "compartment": "WHOLE", "diagnosis": lab.values, "diagnosis_raw": dx.values,
                      "age": get("age").values, "gender": get("gender").values},
                     index=pd.Index("GSE45980|" + m.geo_accession, name="sample_uid"))
    e = e[m.geo_accession]
    e.columns = s.index
    return e, s
