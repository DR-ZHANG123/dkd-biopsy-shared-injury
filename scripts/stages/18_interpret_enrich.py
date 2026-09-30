"""Stage 18e：关键基因的通路富集与「已知 DKD 生物学 vs 新候选」标注。

基因集库：Enrichr 文本库（config interpret.genesets；首次运行下载到 data/external/genesets/，记录 md5 与下载日期
于 manifest.tsv，之后只读本地）。背景 = 基因宇宙（results/04_ranks/gene_universe.tsv），集合先与宇宙取交集并按
min_size / max_size 过滤。
  ORA：每个 (区室, head) 的关键基因分上调 / 下调（e_full 符号）及 top100 上 / 下，超几何检验，库内 BH。
  排序检验：全宇宙 e_full 上，集合内 vs 集合外 Mann–Whitney（效应 = AUC − 0.5，正 = 集合整体推高该病种分数），库内 BH。
已知 DKD 标注（DKD head 关键基因）：DisGeNET / Jensen DISEASES 中匹配 disease_terms_regex 的条目，
  以及 metadata/published_dkd_signatures.tsv（stage 12 审计的已发表签名）。都不含者标为 novel_candidate。
输出 results/18_interpret/enrich/。
"""
from __future__ import annotations

import datetime
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.kpmp_stats import bh  # noqa: E402
from lib.m2_data import universe  # noqa: E402
from lib.repro import ROOT, load_config, md5, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
GS = CFG["interpret"]["genesets"]
W = ROOT / "results/18_interpret/weights"
OUT = ROOT / "results/18_interpret/enrich"
CACHE = ROOT / GS["cache"]


def fetch(lib: str) -> Path:
    p = CACHE / f"{lib}.txt"
    man = CACHE / "manifest.tsv"
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GS["url"] + lib, p)
        rec = pd.DataFrame([{"library": lib, "url": GS["url"] + lib, "downloaded": datetime.date.today().isoformat(),
                             "md5": md5(p), "n_bytes": p.stat().st_size}])
        old = pd.read_csv(man, sep="\t") if man.exists() else pd.DataFrame()
        pd.concat([old[old.library != lib] if len(old) else old, rec]).to_csv(man, sep="\t", index=False)
    return p


def read_lib(p: Path) -> dict[str, set]:
    out = {}
    for line in p.read_text().splitlines():
        f = line.split("\t")
        if len(f) > 2:
            out[f[0]] = {g.split(",")[0].strip() for g in f[2:] if g.strip()}
    return out


def ora(q: set, sets: dict[str, set], U: set) -> pd.DataFrame:
    N, n = len(U), len(q)
    rows = []
    for t, s in sets.items():
        k = len(q & s)
        if k == 0:
            continue
        rows.append({"term": t, "set_size": len(s), "overlap": k, "genes": ";".join(sorted(q & s)),
                     "fold": (k / n) / (len(s) / N), "p": stats.hypergeom.sf(k - 1, N, len(s), n)})
    t = pd.DataFrame(rows)
    if len(t):
        t["fdr"] = bh(t.p.values)
    return t


def ranked(e: pd.Series, sets: dict[str, set]) -> pd.DataFrame:
    r = e.rank().to_numpy()
    idx = {g: i for i, g in enumerate(e.index)}
    n = len(e)
    rows = []
    for t, s in sets.items():
        ii = np.array([idx[g] for g in s if g in idx])
        m = len(ii)
        U1 = r[ii].sum() - m * (m + 1) / 2
        auc = U1 / (m * (n - m))
        sd = np.sqrt(m * (n - m) * (n + 1) / 12)
        z = (U1 - m * (n - m) / 2) / sd
        rows.append({"term": t, "set_size": m, "auc_minus_half": auc - .5, "z": z, "p": 2 * stats.norm.sf(abs(z))})
    t = pd.DataFrame(rows)
    t["fdr"] = bh(t.p.values)
    return t


def known_dkd(U: set) -> dict[str, set]:
    rx = re.compile(GS["disease_terms_regex"])
    out = {}
    for lib in GS["disease_libraries"]:
        for t, s in read_lib(fetch(lib)).items():
            if rx.search(t):
                out[f"{lib}:{t}"] = s & U
    pub = pd.read_csv(ROOT / "metadata/published_dkd_signatures.tsv", sep="\t", dtype=str)
    out["published_signatures(stage12)"] = {g.strip() for x in pub.genes.dropna() for g in x.split(";") if g.strip()} & U
    return out


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    U = set(universe())
    libs = {lib: {t: s & U for t, s in read_lib(fetch(lib)).items()} for lib in GS["libraries"]}
    libs = {lib: {t: s for t, s in d.items() if GS["min_size"] <= len(s) <= GS["max_size"]} for lib, d in libs.items()}
    G = pd.read_csv(W / "gene_effects.tsv.gz", sep="\t")
    O, Rk = [], []
    for (comp, head), g in G.groupby(["compartment", "head"]):
        g = g.set_index("gene")
        e = g.e_full
        top = e.abs().nlargest(100).index
        queries = {"key_up": set(g.index[g.is_key & (e > 0)]), "key_down": set(g.index[g.is_key & (e < 0)]),
                   "top100_up": set(top[e[top] > 0]), "top100_down": set(top[e[top] < 0])}
        for lib, sets in libs.items():
            for qn, q in queries.items():
                if len(q) >= 3:
                    t = ora(q, sets, U)
                    if len(t):
                        O.append(t.assign(compartment=comp, head=head, library=lib, query=qn, n_query=len(q)))
            Rk.append(ranked(e, sets).assign(compartment=comp, head=head, library=lib))
    O, Rk = pd.concat(O, ignore_index=True), pd.concat(Rk, ignore_index=True)
    kd = known_dkd(U)
    D = G[(G["head"] == "DKD") & G.is_key].copy()
    for name, s in kd.items():
        D[name] = D.gene.isin(s)
    src = list(kd)
    D["known_sources"] = D[src].apply(lambda r: ";".join(c for c in src if r[c]), axis=1)
    D["annotation"] = np.where(D[src].any(axis=1), "known_DKD", "novel_candidate")
    D = D.drop(columns=src)
    files = {"ora.tsv": O, "ranked.tsv": Rk, "dkd_key_known_vs_novel.tsv": D}
    for f, t in files.items():
        t.to_csv(OUT / f, sep="\t", index=False)
    man = pd.read_csv(CACHE / "manifest.tsv", sep="\t")
    write_provenance("18_interpret/enrich", [W / "gene_effects.tsv.gz", ROOT / "metadata/published_dkd_signatures.tsv"]
                     + [CACHE / f"{lib}.txt" for lib in GS["libraries"] + GS["disease_libraries"]],
                     [OUT / f for f in files], CFG["seed"],
                     {"genesets": man.to_dict(orient="records"),
                      "known_sets_sizes": {k: len(v) for k, v in kd.items()}})
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        d = O[(O["head"] == "DKD") & (O.fdr < GS["fdr"])].sort_values("p")
        print(d[["compartment", "library", "query", "term", "overlap", "fold", "fdr"]].head(40).to_string())
        r = Rk[(Rk["head"] == "DKD") & (Rk.fdr < GS["fdr"])].sort_values("p")
        print(r[["compartment", "library", "term", "set_size", "auc_minus_half", "fdr"]].head(40).to_string())
        print(D[["compartment", "gene", "e_full", "annotation", "known_sources"]].to_string())


if __name__ == "__main__":
    main()
