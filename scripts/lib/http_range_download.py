"""并行 HTTP range 下载（CELLxGENE CDN 单连接限速约 2 MB/s 时使用；断点续传）。

用法: python scripts/lib/http_range_download.py URL OUT [n_threads=32] [chunk_MB=32]
"""
import os, sys, requests, concurrent.futures as cf
url, out = sys.argv[1], sys.argv[2]
nt = int(sys.argv[3]) if len(sys.argv) > 3 else 32
ch = (int(sys.argv[4]) if len(sys.argv) > 4 else 32) << 20
size = int(requests.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])
part = out + ".part"
if not os.path.exists(part) or os.path.getsize(part) != size:
    with open(part, "wb") as fh: fh.truncate(size)
done_f = part + ".done"
done = set(map(int, open(done_f).read().split())) if os.path.exists(done_f) else set()
def get(i):
    a, b = i * ch, min(size, (i + 1) * ch) - 1
    for _ in range(20):
        try:
            r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=300)
            if r.status_code == 206 and len(r.content) == b - a + 1:
                with open(part, "r+b") as fh: fh.seek(a); fh.write(r.content)
                return i
        except Exception: pass
    raise RuntimeError(f"chunk {i} failed")
todo = [i for i in range((size + ch - 1) // ch) if i not in done]
with cf.ThreadPoolExecutor(nt) as ex, open(done_f, "a") as dfh:
    for k, i in enumerate(ex.map(get, todo)):
        dfh.write(f"{i}\n"); dfh.flush()
        if k % 20 == 0: print(f"{k+1}/{len(todo)}", flush=True)
os.rename(part, out); os.remove(done_f); print("ok", size)
