"""Stage 01：解析全部 series matrix / RNA-seq counts，probe→Entrez→当前 symbol，检查 deposited 矩阵尺度。

产出：data/interim/<series_key>_expr.parquet（gene × sample，log2 绝对尺度）
      data/interim/<series_key>_meta.tsv（首列 gsm；GEO 样本注释原文，诊断字段不在此解析）
      results/01_ingest/series_scale_audit.tsv、gene_mapping_stats.tsv
平台注释来自 GPL family SOFT 的 platform_table；BrainArray ENTREZG 定制 CDF 直接由 ID 取 Entrez。
GSE30528/GSE30529 的 deposited 矩阵为 probe-centred，改用同项目旧版由 CEL 重做的探针级 RMA
（已按 CEL 文件校正样本对应），再用本 stage 同一套 GPL571 注释折叠到基因。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.geo import (GeneIndex, collapse_to_symbol, platform_entrez,  # noqa: E402
                     read_series_matrix, read_soft_platform_table)
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
RAW = ROOT / CFG["paths"]["raw"]
INTERIM = ROOT / CFG["paths"]["interim"]
OUT = ROOT / "results" / "01_ingest"
OLD_RAW = (ROOT / CFG["old_project_raw"]).resolve()
OLD_PROC = (ROOT / CFG["old_project_processed"]).resolve()
RNASEQ_NCBI = ("GSE142025", "GSE175759")          # NCBI 统一比对 counts，行 = Entrez GeneID
RNASEQ_AUTHOR = "GSE182380"                        # 作者上传 counts，行 = gene symbol
# Illumina WG-DASL：deposited 矩阵已按基因中心化/按基因缩放（见 scale audit），改用 GEO 补充文件
# 中的 non-normalized AVG_Signal；样本内秩对逐样本单调变换（log2、分位数标准化）不变。
# 补充文件的探针行标注用「与 deposited 矩阵逐探针跨样本相关」核对（逐样本单调变换不改变该相关），
# 中位 r < MIN_ALIGN_R 视为行标注错位，整个 series 排除。
ILLUMINA_NONNORM = ("GSE115857", "GSE116626")
MIN_ALIGN_R = 0.9
EXCLUDED_SCALES = ("probe_centred", "gene_scaled", "nonnorm_misaligned")
SOFT_CACHE: dict[str, tuple[pd.DataFrame, str]] = {}


def soft_table(gpl: str) -> tuple[pd.DataFrame, str]:
    if gpl not in SOFT_CACHE:
        SOFT_CACHE[gpl] = read_soft_platform_table(RAW / "series" / f"{gpl}_family.soft.gz")
    return SOFT_CACHE[gpl]


def scale_audit(key: str, expr: pd.DataFrame) -> dict:
    v = expr.values.astype(float)
    row_mean = np.nanmean(v, axis=1)
    return {
        "series": key, "n_samples": expr.shape[1], "n_genes": expr.shape[0],
        "median": float(np.nanmedian(v)), "q01": float(np.nanquantile(v, 0.01)),
        "q99": float(np.nanquantile(v, 0.99)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
        "frac_negative": float(np.mean(v < 0)), "frac_nan": float(np.mean(np.isnan(v))),
        "mean_abs_row_mean": float(np.nanmean(np.abs(row_mean))),
        "sd_row_mean": float(np.nanstd(row_mean)),
        "median_col_sd": float(np.nanmedian(np.nanstd(v, axis=0))),
    }


def classify_scale(a: dict) -> str:
    """gene_centred：行均值在基因间几乎不变（逐基因中心化或逐基因缩放），样本内秩不再反映表达量。
    真实 log2 强度下行均值的离散度与样本内离散度同量级（≈2）。"""
    if a["sd_row_mean"] < 0.25 * a["median_col_sd"]:
        return "probe_centred" if a["mean_abs_row_mean"] < 0.5 else "gene_scaled"
    if a["max"] > 100:
        return "linear"
    return "log2"


def ingest_arrays(gidx: GeneIndex) -> tuple[list[dict], list[dict], list[Path], list[Path]]:
    audits, maps, inputs, outputs = [], [], [], []
    for sm in sorted((RAW / "series").glob("GSE*/*_series_matrix.txt.gz")):
        gse = sm.parent.name
        if gse in RNASEQ_NCBI or gse == RNASEQ_AUTHOR:
            continue
        expr, meta = read_series_matrix(sm)
        deposited_scale, align_r = None, np.nan
        if gse in ILLUMINA_NONNORM:
            deposited_scale = classify_scale(scale_audit(gse, expr))
            nonnorm = illumina_nonnormalized(gse, meta)
            align_r = row_alignment(nonnorm, expr)
            expr = nonnorm
            inputs.append(RAW / "illumina" / f"{gse}_non_normalized.txt.gz")
        gpls = meta["platform_id"].unique()
        assert len(gpls) == 1, f"{sm}: 多平台 {gpls}"
        gpl = gpls[0]
        key = f"{gse}-{gpl}"
        tab, title = soft_table(gpl)
        ent, col = platform_entrez(tab, expr.index)
        gene, st = collapse_to_symbol(expr, ent, gidx)
        a = scale_audit(key, gene)
        a["scale"] = classify_scale(a)
        a["nonnorm_vs_deposited_gene_r"] = align_r
        if align_r < MIN_ALIGN_R:
            a["scale"] = "nonnorm_misaligned"
        if a["scale"] == "nonnorm_misaligned":
            a["action"] = (f"excluded: deposited matrix {deposited_scale}; non-normalized table rows disagree "
                           f"with deposited matrix (median per-probe r = {align_r:.2f})")
        elif a["scale"] == "linear":
            gene = np.log2(gene.clip(lower=1))
            a["action"] = "log2(max(x,1))"
            if deposited_scale is not None:
                a["action"] = (f"deposited matrix {deposited_scale}; replaced by GEO non-normalized "
                               "AVG_Signal, log2(max(x,1))")
        elif a["scale"] in ("probe_centred", "gene_scaled"):
            a["action"] = "excluded: within-sample ranks undefined for gene-centred values"
        else:
            a["action"] = "none"
        a.update(technology="array", gpl=gpl, source=sm.name, deposited_scale=deposited_scale or a["scale"])
        audits.append(a)
        maps.append({"series": key, "gpl": gpl, "gpl_title": title, "id_column": col, **st})
        inputs += [sm, RAW / "series" / f"{gpl}_family.soft.gz"]
        if a["scale"] not in EXCLUDED_SCALES:
            write_series(key, gene, meta.rename(columns={"geo_accession": "gsm"}), outputs)
        print(key, gene.shape, a["scale"], col, flush=True)
    return audits, maps, inputs, outputs


def row_alignment(nonnorm: pd.DataFrame, deposited: pd.DataFrame) -> float:
    """同一探针 ID 下 log2(non-normalized) 与 deposited 值的跨样本 Pearson 相关的中位数。"""
    x = np.log2(nonnorm.clip(lower=1)).reindex(index=deposited.index, columns=deposited.columns)
    xa = x.sub(x.mean(1), axis=0)
    da = deposited.sub(deposited.mean(1), axis=0)
    r = (xa * da).sum(1) / np.sqrt((xa ** 2).sum(1) * (da ** 2).sum(1))
    return float(np.nanmedian(r))


def illumina_nonnormalized(gse: str, meta: pd.DataFrame) -> pd.DataFrame:
    """non-normalized 表的列为芯片条码（= !Sample_description），转成 GSM 列名。"""
    raw = pd.read_csv(RAW / "illumina" / f"{gse}_non_normalized.txt.gz", sep="\t", index_col=0)
    sig = raw.filter(regex=r"AVG_Signal")
    sig.columns = sig.columns.str.replace(r"^AVG_Signal-|\.AVG_Signal$", "", regex=True)
    bc2gsm = dict(zip(meta["description"], meta["geo_accession"]))
    assert set(sig.columns) == set(bc2gsm), f"{gse}: non-normalized 条码与 GEO description 不一致"
    sig = sig.rename(columns=bc2gsm)
    sig.index = sig.index.astype(str)
    return sig.apply(pd.to_numeric, errors="coerce")


def write_series(key: str, gene: pd.DataFrame, meta: pd.DataFrame, outputs: list[Path]) -> None:
    meta = meta[["gsm"] + [c for c in meta.columns if c != "gsm"]]
    assert set(gene.columns) == set(meta.gsm), f"{key}: 表达列与 meta 的 GSM 不一致"
    gene = gene[list(meta.gsm)].astype("float32")
    gene.to_parquet(INTERIM / f"{key}_expr.parquet")
    meta.to_csv(INTERIM / f"{key}_meta.tsv", sep="\t", index=False)
    outputs += [INTERIM / f"{key}_expr.parquet", INTERIM / f"{key}_meta.tsv"]


def ingest_rnaseq(gidx: GeneIndex) -> tuple[list[dict], list[dict], list[Path], list[Path]]:
    audits, maps, inputs, outputs = [], [], [], []
    for g in (*RNASEQ_NCBI, RNASEQ_AUTHOR):
        sm = next((RAW / "series" / g).glob("*_series_matrix.txt.gz"))
        _, meta = read_series_matrix(sm)
        meta = meta.rename(columns={"geo_accession": "gsm"})
        if g == RNASEQ_AUTHOR:
            src = RAW / "rnaseq" / "GSE182380_NEPTUNE_TI_adult_RNA_counts_2021.txt.gz"
            counts = pd.read_csv(src, sep="\t", index_col=0)
            # 列名 Sample_N ↔ series matrix 标题 "Sample_N RNAseq counts from ..."
            title2gsm = dict(zip(meta.title.str.split().str[0], meta.gsm))
            assert set(counts.columns) == set(title2gsm), "GSE182380：counts 列名与 GEO 标题不能一一对应"
            counts.columns = [title2gsm[c] for c in counts.columns]
            ent = pd.Series([gidx.symbol_to_entrez(s) for s in counts.index.astype(str)], index=counts.index)
            id_col = "GeneSymbol→Entrez(symbol/unique synonym)"
        else:
            src = RAW / "rnaseq" / f"{g}_raw_counts_GRCh38.p13_NCBI.tsv.gz"
            counts = pd.read_csv(src, sep="\t", index_col=0)
            counts.index = counts.index.astype(str)
            ent = pd.Series(counts.index, index=counts.index)
            id_col = "GeneID"
        sym = ent.map(lambda e: gidx.e2s.get(e) if isinstance(e, str) else None)
        cpm = counts / counts.sum(axis=0) * 1e6          # 文库大小 = 映射前全部行之和
        cpm = cpm.loc[sym.notna().values]
        cpm.index = sym[sym.notna()].values
        # 同一 Entrez 的多行（作者表里的旧 symbol / 别名）在计数尺度上求和
        gene = np.log2(cpm.groupby(level=0).sum() + 1).rename_axis("gene")
        st = {"n_probes": int(len(counts)), "n_probes_entrez": int(ent.notna().sum()),
              "n_probes_symbol": int(sym.notna().sum()), "n_genes": int(len(gene))}
        key = f"{g}-RNAseq"
        a = scale_audit(key, gene)
        a.update(scale="log2cpm", action="log2(CPM+1)", technology="rnaseq",
                 gpl=meta.platform_id.iloc[0], source=src.name)
        audits.append(a)
        maps.append({"series": key, "gpl": a["gpl"], "gpl_title": "RNA-seq counts", "id_column": id_col, **st})
        inputs += [src, sm]
        write_series(key, gene, meta, outputs)
        print(key, gene.shape, flush=True)
    return audits, maps, inputs, outputs


def ingest_cel_reprocessed(gidx: GeneIndex) -> tuple[list[dict], list[dict], list[Path], list[Path]]:
    audits, maps, inputs, outputs = [], [], [], []
    tab, title = soft_table("GPL571")
    for cohort, gse in (("GLOM_GSE30528", "GSE30528"), ("TUB_GSE30529", "GSE30529")):
        probe_f = OLD_PROC / "bulk" / f"{gse}_rma_probe.tsv.gz"
        probe = pd.read_csv(probe_f, sep="\t", index_col=0)
        meta = pd.read_csv(OLD_PROC / "bulk" / f"{cohort}_meta.tsv", sep="\t", dtype=str)
        ent, col = platform_entrez(tab, probe.index)
        gene, st = collapse_to_symbol(probe, ent, gidx)
        # 一致性核对：旧版基因矩阵（旧 symbol）与本次折叠在共有 symbol 上逐样本相关应≈1
        old = pd.read_csv(OLD_PROC / "bulk" / f"{cohort}_gene.tsv.gz", sep="\t", index_col=0)
        common = gene.index.intersection(old.index)
        r = [np.corrcoef(gene.loc[common, c], old.loc[common, c])[0, 1] for c in gene.columns]
        assert min(r) > 0.99, f"{gse}: 与旧版 CEL 基因矩阵列对应不一致 (min r={min(r):.3f})"
        key = f"{gse}-GPL571CEL"
        a = scale_audit(key, gene)
        a.update(scale="log2_rma_from_cel", action="deposited matrix probe-centred; CEL RMA used",
                 technology="array", gpl="GPL571", source=probe_f.name)
        audits.append(a)
        maps.append({"series": key, "gpl": "GPL571", "gpl_title": title, "id_column": col, **st,
                     "min_r_vs_old_gene_matrix": float(min(r))})
        inputs += [probe_f, OLD_PROC / "bulk" / f"{cohort}_meta.tsv"]
        write_series(key, gene, meta, outputs)
        print(key, gene.shape, f"min r vs old={min(r):.4f}", flush=True)
    return audits, maps, inputs, outputs


def main() -> None:
    set_global_seed(CFG["seed"])
    INTERIM.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    gene_info = OLD_RAW / "metadata" / "Homo_sapiens.gene_info.gz"
    gidx = GeneIndex(gene_info)
    audits, maps, inputs, outputs = [], [], [gene_info], []
    for fn in (ingest_arrays, ingest_rnaseq, ingest_cel_reprocessed):
        a, m, i, o = fn(gidx)
        audits += a
        maps += m
        inputs += i
        outputs += o
    aud = pd.DataFrame(audits)
    aud.to_csv(OUT / "series_scale_audit.tsv", sep="\t", index=False)
    pd.DataFrame(maps).to_csv(OUT / "gene_mapping_stats.tsv", sep="\t", index=False)
    bad = aud.scale.isin(EXCLUDED_SCALES)
    kept = aud[~bad]
    write_provenance("01_ingest", sorted(set(inputs)),
                     outputs + [OUT / "series_scale_audit.tsv", OUT / "gene_mapping_stats.tsv"], CFG["seed"],
                     {"n_series": int(len(aud)), "n_series_kept": int(len(kept)),
                      "n_samples_kept": int(kept.n_samples.sum()),
                      "excluded_series": aud.loc[bad, "series"].tolist()})
    print(aud.to_string())


if __name__ == "__main__":
    main()
