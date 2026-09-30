"""稿件图文件核对：每个 figure 环境按出现顺序编号（主图 1..n，补充图 S1..Sm），
其 \\includegraphics 文件名必须是对应编号，且无文件被重复引用；对应的绘图脚本必须存在。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tex = (ROOT / "manuscript/humgenomics/main.tex").read_text() if len(sys.argv) < 2 else Path(sys.argv[1]).read_text()
envs = re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", tex, flags=re.S)
errs, seen, n_main, n_supp = [], set(), 0, 0
for body in envs:
    files = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
    lab = re.search(r"\\label\{([^}]+)\}", body)
    lab = lab.group(1) if lab else "?"
    supp = lab.startswith("figS")
    if supp:
        n_supp += 1; want = f"FigS{n_supp}"
    else:
        n_main += 1; want = f"Fig{n_main}"
    for f in files:
        stem = Path(f).stem
        if stem != want:
            errs.append(f"{lab}: 第 {want} 个图却引用 {stem}")
        if stem in seen:
            errs.append(f"{stem} 被重复引用")
        seen.add(stem)
        if not list((ROOT / "scripts/figures").glob(f"{stem}_*.py")):
            errs.append(f"{stem} 找不到绘图脚本 scripts/figures/{stem}_*.py")
print(f"main {n_main}, supplementary {n_supp}")
if errs:
    print("\n".join(errs)); sys.exit(1)
print("FIGURE FILES OK")
