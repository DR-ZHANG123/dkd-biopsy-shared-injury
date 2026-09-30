#!/usr/bin/env bash
# 独立测试集候选（stage 17）：
# 1) KPMP Kidney Tissue Atlas repository 开放获取层：区域（激光显微切割，LMD）转录组 + 活检切片 bulk total/mRNA。
#    来源：https://atlas.kpmp.org/repository （experimental_strategy = "Regional Transcriptomics" / "Bulk Total/mRNA"）
#    只取 access = open（非标准化计数 csv、实验元数据 xlsx、Atlas Explorer 汇总包）；fastq/bam 为 controlled，不下载。
# 2) 公开 GEO 候选（KPMP LMD 样本量不足时的备选；只下载，stage 17a 只做样本构成与查重，不做任何模型评估）：
#    GSE162830 LMD 肾小球 RNA-seq（DN / 特发性结节性肾小球硬化 / 参照肾切除）
#    GSE166239 FFPE 全切片活检 RNA-seq（T2DN / 高血压肾病 / 对照）
#    GSE45980、GSE60860 冷冻切片活检 Agilent 芯片（多种蛋白尿肾病，含 DN）
# 大文件落在 data/raw/kpmp_lmd 与 data/raw/indep（data/ (local data directory)）。幂等。
set -euo pipefail
cd "$(dirname "$0")/../.."
RAW=data/raw/kpmp_lmd; mkdir -p "$RAW" results/00_download
python scripts/lib/kpmp_repo.py "$RAW" "Regional Transcriptomics" "Bulk Total/mRNA"
cp "$RAW/checksums.tsv" results/00_download/kpmp_lmd_checksums.tsv
cp "$RAW"/manifest_*.tsv results/00_download/

IND=data/raw/indep; mkdir -p "$IND"
MAN=results/00_download/indep_candidates_checksums.tsv; : > "$MAN.tmp"
fetch() { local url=$1 out=$2; [[ -s $out ]] || { curl -fsSL --retry 5 -o "$out.part" "$url" && mv "$out.part" "$out"; }; echo -e "$(basename "$out")\t$(md5sum "$out" | cut -d' ' -f1)\t$url" >> "$MAN.tmp"; }
geo() { local g=$1; echo "https://ftp.ncbi.nlm.nih.gov/geo/series/${g:0:${#g}-3}nnn/$g"; }
for g in GSE162830 GSE166239 GSE45980 GSE60860; do
  for f in $(curl -fsSL "$(geo $g)/matrix/" | grep -o 'GSE[0-9]*[-A-Z0-9]*_series_matrix.txt.gz' | sort -u); do
    fetch "$(geo $g)/matrix/$f" "$IND/$f"
  done
done
fetch "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE162830&format=file&file=GSE162830_raw_counts_GRCh38.p13_NCBI.tsv.gz" "$IND/GSE162830_raw_counts_GRCh38.p13_NCBI.tsv.gz"
fetch "$(geo GSE162830)/suppl/GSE162830_ING_quantile_normalized_final.csv.gz" "$IND/GSE162830_ING_quantile_normalized_final.csv.gz"
fetch "$(geo GSE166239)/suppl/GSE166239_Nordbo_et_al_counts.txt.gz" "$IND/GSE166239_Nordbo_et_al_counts.txt.gz"
fetch "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13497/soft/GPL13497_family.soft.gz" "$IND/GPL13497_family.soft.gz"
mv "$MAN.tmp" "$MAN"
echo "done: KPMP $(($(wc -l < results/00_download/kpmp_lmd_checksums.tsv) - 1)) open files; GEO $(wc -l < "$MAN") files"
