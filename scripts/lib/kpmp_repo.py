"""KPMP Kidney Tissue Atlas repository（atlas.kpmp.org/repository）开放获取层的文件清单与下载。

清单来自 repository 网页前端使用的公开搜索端点（Elastic App Search，engine = atlas-repository；
检索用的 public search key 写在网页 JS 中，只读）。下载端点 = /api/v1/file/download/<package_id>/<file_name>。
只下载 access == open 的文件；controlled 层（fastq/bam）不下载。

用法：python scripts/lib/kpmp_repo.py <out_dir> "Regional Transcriptomics" "Bulk Total/mRNA"
产出：<out_dir>/manifest_<strategy>.tsv（全部文件的元数据，含 controlled，用于样本构成统计）、
      <out_dir>/files/<file_name>（open 文件）、<out_dir>/checksums.tsv（md5、URL、大小）。
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

SEARCH = "https://atlas.kpmp.org/spatial-viewer/search/api/as/v1/engines/atlas-repository/search.json"
SEARCH_KEY = "search-vwz67uj2sf8h83h4y8i8j6g3"       # 网页前端公开的只读检索 key
DOWNLOAD = "https://atlas.kpmp.org/api/v1/file/download"


def listing(strategy: str) -> pd.DataFrame:
    rows, page = [], 1
    while True:
        r = requests.post(SEARCH, json={"query": "", "page": {"size": 1000, "current": page},
                                        "filters": {"all": [{"experimental_strategy": strategy}]}},
                          headers={"Authorization": f"Bearer {SEARCH_KEY}"}, timeout=120)
        r.raise_for_status()
        js = r.json()
        rows += [{k: v["raw"] for k, v in x.items() if k != "_meta"} for x in js["results"]]
        if page >= js["meta"]["page"]["total_pages"]:
            break
        page += 1
    d = pd.DataFrame(rows)
    for c in d.columns:
        d[c] = d[c].map(lambda x: ";".join(map(str, x)) if isinstance(x, list) else x)
    return d.sort_values("file_name").reset_index(drop=True)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(package_id: str, file_name: str, out: Path, size: float | None) -> str:
    url = f"{DOWNLOAD}/{package_id}/{urllib.parse.quote(file_name)}"
    if out.exists() and (size is None or out.stat().st_size == int(size)):
        return url
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp = out.with_suffix(out.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for b in r.iter_content(1 << 20):
                        fh.write(b)
            if size is not None and tmp.stat().st_size != int(size):
                raise IOError(f"size mismatch {tmp.stat().st_size} != {int(size)}")
            tmp.rename(out)
            return url
        except Exception as e:  # noqa: BLE001
            print(f"retry {attempt + 1}: {file_name}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"下载失败：{url}")


def main() -> None:
    out = Path(sys.argv[1])
    (out / "files").mkdir(parents=True, exist_ok=True)
    ck = []
    for strat in sys.argv[2:]:
        d = listing(strat)
        tag = strat.split()[0].split("/")[0].lower()
        d.to_csv(out / f"manifest_{tag}.tsv", sep="\t", index=False)
        op = d[d.access == "open"]
        print(f"{strat}: {len(d)} files, open {len(op)}", flush=True)
        for _, r in op.iterrows():
            f = out / "files" / r.file_name
            url = fetch(r.package_id, r.file_name, f, r.file_size)
            ck.append({"file": r.file_name, "md5": md5(f), "bytes": f.stat().st_size, "url": url,
                       "strategy": strat, "workflow_type": r.workflow_type, "redcap_id": r.redcap_id})
    ck = pd.DataFrame(ck).drop_duplicates("file")
    ck.to_csv(out / "checksums.tsv", sep="\t", index=False)
    print(f"done: {len(ck)} open files", flush=True)


if __name__ == "__main__":
    main()
