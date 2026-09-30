#!/usr/bin/env bash
# KPMP 人肾单细胞 / 单核参照（CZ CELLxGENE Discover 公开 h5ad）。幂等：已存在的文件跳过。
# Collection：Cellular and Spatial Drivers of Unresolved Injury and Functional Decline in the Human Kidney
#   collection_id 9c9d04c4-8899-417f-bb6f-6107dcadf14f（KPMP atlas v2.0；v1 = Lake et al. 2023 Nature，collection bcb61471-...）
# 大文件落在 data/raw/kpmp（data/ (local data directory)）。
set -euo pipefail
cd "$(dirname "$0")/../.."
RAW=data/raw/kpmp; mkdir -p "$RAW" results/00_download
MAN=results/00_download/kpmp_checksums.tsv; : > "$MAN.tmp"
API=https://api.cellxgene.cziscience.com/curation/v1/collections/9c9d04c4-8899-417f-bb6f-6107dcadf14f
[[ -s $RAW/collection_9c9d04c4.json ]] || curl -fsSL --retry 5 -o "$RAW/collection_9c9d04c4.json" "$API"
# dataset_version_id（= 文件名）  dataset_id  标题
for spec in "06b8d9c2-0bde-48bd-88d9-730993fc95e2 7ff0197b-d175-49bf-b4fa-150fe0995d93 snRNA_v2.0" \
            "255ce81a-eb4e-48ff-9f18-e50455ccb24d 91f31e05-56d8-46fc-b408-d90c9228a81b scRNA_v2.0"; do
  read -r ver did label <<< "$spec"
  out=$RAW/$ver.h5ad; url=https://datasets.cellxgene.cziscience.com/$ver.h5ad
  [[ -s $out ]] || python scripts/lib/http_range_download.py "$url" "$out" 32 32
  # 完整性：与 CDN 的 S3 multipart ETag（8 MiB 分片 md5 的 md5）比对
  etag=$(curl -fsSI "$url" | tr -d '\r' | awk -F'"' 'tolower($0) ~ /^etag/{print $2}')
  python - "$out" "$etag" <<'PY'
import hashlib, sys
f, et = sys.argv[1], sys.argv[2]
n = int(et.split("-")[1]) if "-" in et else 1
ps = 8 << 20; ds = []
with open(f, "rb") as fh:
    for b in iter(lambda: fh.read(ps), b""):
        ds.append(hashlib.md5(b).digest())
got = hashlib.md5(b"".join(ds)).hexdigest() + f"-{len(ds)}" if n > 1 else hashlib.md5(open(f, "rb").read()).hexdigest()
assert got == et, f"ETag mismatch {f}: {got} != {et}"
print("etag ok", f)
PY
  echo -e "$(basename "$out")\t$(md5sum "$out" | cut -d' ' -f1)\t$url\t$did\t$label\tetag=$etag" >> "$MAN.tmp"
done
echo -e "collection_9c9d04c4.json\t$(md5sum "$RAW/collection_9c9d04c4.json" | cut -d' ' -f1)\t$API\t-\tcollection_metadata" >> "$MAN.tmp"
mv "$MAN.tmp" "$MAN"; echo "done: $(wc -l < "$MAN") files"
