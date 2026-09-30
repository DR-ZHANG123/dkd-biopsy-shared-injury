"""Stage 24a：损伤–修复反应的通路注释（预排序 GSEA）。

排序统计量 = stage 19 反应 meta 分析的随机效应 z（非 DKD 病人 vs 对照；GLOM 与 TUB 分别）。
基因集 = MSigDB 2024.1.Hs Hallmark（主）、Reactome、GO BP（config pathways_sensitivity.gsea）。
两个基因变体：full（全部基因）与 no_proc_ieg（剔除取材敏感基因与即早基因，规则同 stage 21 A3 敏感性分析）。
产出 results/24_pathways_sensitivity/gsea/gsea_<compartment>_<library>.tsv（NES、nominal P、FDR、leading edge）
与 gsea_summary.tsv（每个区室 × 库 FDR < 0.05 的上 / 下调通路数与前 10 条）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import gseapy as gp
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from lib.robustness21 import exclude  # noqa: E402

CFG = load_config()
G = CFG["pathways_sensitivity"]["gsea"]
OUT = ROOT / "results" / "24_pathways_sensitivity" / "gsea"


def run(comp: str, lib: str, gmt: Path, variant: str) -> pd.DataFrame:
    m = pd.read_csv(ROOT / f"results/19_shared_program/core/meta_{comp}.tsv.gz", sep="\t")
    rnk = m[["gene", G["rank_by"]]].dropna().drop_duplicates("gene").sort_values(G["rank_by"], ascending=False)
    if variant != "full":
        rnk = rnk[~rnk.gene.isin(exclude(variant, rnk.set_index("gene")[G["rank_by"]]))]
    res = gp.prerank(rnk=rnk.set_index("gene")[G["rank_by"]], gene_sets=str(gmt), min_size=G["min_size"],
                     max_size=G["max_size"], permutation_num=G["permutations"], seed=CFG["seed"], threads=8,
                     outdir=None, verbose=False).res2d
    res = res.rename(columns={"Term": "term", "NES": "nes", "NOM p-val": "p", "FDR q-val": "fdr",
                              "Lead_genes": "leading_edge", "Tag %": "tag_frac"})
    res = res[["term", "nes", "p", "fdr", "tag_frac", "leading_edge"]].assign(compartment=comp, library=lib,
                                                                               variant=variant, n_ranked=len(rnk))
    for c in ("nes", "p", "fdr"):
        res[c] = res[c].astype(float)
    return res.sort_values("nes", ascending=False)


def main() -> None:
    set_global_seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    outs, summ = [], []
    for variant, comp in [(v, c) for v in ("full", "no_proc_ieg") for c in ("GLOM", "TUB")]:
        for lib, gmt in G["libraries"].items():
            r = run(comp, lib, ROOT / gmt, variant)
            p = OUT / f"gsea_{comp}_{lib}{'' if variant == 'full' else '_' + variant}.tsv"
            r.to_csv(p, sep="\t", index=False)
            outs.append(p)
            sig = r[r.fdr < G["fdr"]]
            for d, s in (("up", sig[sig.nes > 0]), ("down", sig[sig.nes < 0].sort_values("nes"))):
                summ.append({"variant": variant, "compartment": comp, "library": lib, "direction": d, "n_sets_tested": len(r),
                             "n_fdr05": len(s), "top10": "; ".join(s.term.head(10))})
    pd.DataFrame(summ).to_csv(OUT / "gsea_summary.tsv", sep="\t", index=False)
    outs.append(OUT / "gsea_summary.tsv")
    ins = [ROOT / f"results/19_shared_program/core/meta_{c}.tsv.gz" for c in ("GLOM", "TUB")]
    ins += [ROOT / g for g in G["libraries"].values()]
    write_provenance("24_pathways_sensitivity/gsea", ins, outs, CFG["seed"], {"gseapy": gp.__version__})
    print(pd.DataFrame(summ)[["variant", "compartment", "library", "direction", "n_sets_tested", "n_fdr05"]].to_string())


if __name__ == "__main__":
    main()
