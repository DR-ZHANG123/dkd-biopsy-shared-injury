#!/usr/bin/env bash
# 下载全部公开数据并记录 md5。幂等：已存在且非空的文件跳过。
set -euo pipefail
cd "$(dirname "$0")/../.."
RAW=data/raw; mkdir -p "$RAW"/{series,rnaseq,string} results/00_download
MAN=results/00_download/checksums.tsv; : > "$MAN.tmp"
fetch() { local url=$1 out=$2; [[ -s $out ]] || { curl -fsSL --retry 5 -o "$out.part" "$url" && mv "$out.part" "$out"; }; echo -e "$(basename "$out")\t$(md5sum "$out" | cut -d' ' -f1)\t$url" >> "$MAN.tmp"; }
series_dir() { local g=$1; echo "https://ftp.ncbi.nlm.nih.gov/geo/series/${g:0:${#g}-3}nnn/$g/matrix/"; }
# 所有芯片 series：列出 matrix 目录下全部 *_series_matrix.txt.gz（多平台 series 有多个文件）
for g in GSE96804 GSE99339 GSE104948 GSE104954 GSE47183 GSE37460 GSE32591 GSE93798 GSE69438 GSE116626 GSE115857 GSE108109 GSE108112 GSE133288 GSE200818 GSE142025 GSE175759 GSE182380; do
  mkdir -p "$RAW/series/$g"
  for f in $(curl -fsSL "$(series_dir $g)" | grep -o 'GSE[0-9]*[-A-Z0-9]*_series_matrix.txt.gz' | sort -u); do
    fetch "$(series_dir $g)$f" "$RAW/series/$g/$f"
  done
done
# RNA-seq：NCBI 统一比对的原始 counts（GRCh38）与注释
for g in GSE142025 GSE175759; do
  fetch "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=$g&format=file&file=${g}_raw_counts_GRCh38.p13_NCBI.tsv.gz" "$RAW/rnaseq/${g}_raw_counts_GRCh38.p13_NCBI.tsv.gz"
done
# NCBI counts 的行即 Entrez GeneID；symbol 由 NCBI gene_info 统一转换（stage 01），不再下载 GEO 注释表
# （该地址对脚本返回验证页而非文件）。
# GSE182380 无 NCBI 统一 counts，使用作者上传的 counts 表
fetch https://ftp.ncbi.nlm.nih.gov/geo/series/GSE182nnn/GSE182380/suppl/GSE182380_NEPTUNE_TI_adult_RNA_counts_2021.txt.gz "$RAW/rnaseq/GSE182380_NEPTUNE_TI_adult_RNA_counts_2021.txt.gz"
# Illumina WG-DASL：deposited 矩阵已逐基因中心化/缩放，使用 GEO 补充文件中的 non-normalized 信号（stage 01 核对行标注）
mkdir -p "$RAW/illumina"
for g in GSE115857 GSE116626; do
  fetch "https://ftp.ncbi.nlm.nih.gov/geo/series/${g:0:${#g}-3}nnn/$g/suppl/${g}_non_normalized.txt.gz" "$RAW/illumina/${g}_non_normalized.txt.gz"
done
# STRING v12
fetch https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz "$RAW/string/9606.protein.links.v12.0.txt.gz"
fetch https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz "$RAW/string/9606.protein.info.v12.0.txt.gz"
# 平台注释（GPL 全表）
for gpl in GPL11670 GPL14663 GPL14951 GPL19983 GPL22945 GPL24120 GPL19184 GPL19109 GPL17586 GPL571; do
  fetch "https://ftp.ncbi.nlm.nih.gov/geo/platforms/${gpl:0:${#gpl}-3}nnn/$gpl/soft/${gpl}_family.soft.gz" "$RAW/series/${gpl}_family.soft.gz"
done
# MSigDB 2024.1.Hs 基因集（stage 24a 通路注释；落在 data/external/msigdb，与 config pathways_sensitivity.gsea 一致）
mkdir -p data/external/msigdb
for f in h.all c2.cp.reactome c5.go.bp; do
  fetch "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/$f.v2024.1.Hs.symbols.gmt" "data/external/msigdb/$f.v2024.1.Hs.symbols.gmt"
done
mv "$MAN.tmp" "$MAN"; echo "done: $(wc -l < "$MAN") files"
