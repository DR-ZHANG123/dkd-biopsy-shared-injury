"""Stage 16a：反卷积输入 —— KPMP 细胞类型 / 状态参照（snRNA 主、scRNA 敏感性；GLOM / TUB 两套）与 bulk 混合物。

样本 = stage 11 全部单元成员 ∪ keep_for_pretrain（去重样本；含外部测试队列，反卷积不读诊断字段）。
每个 (series, 区室) 写一个混合物文件。输出（大文件）在 config deconv.out_dir；摘要表在 results/16_deconv/prepare/。
用法：python scripts/stages/16_deconv_prepare.py [--references snRNA,scRNA] [--skip-mixtures]
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.deconv_ref import (DC, OUT, bulk_gene_union, build_reference, mixture_groups,  # noqa: E402
                            reference_summary, write_mixture)
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

CFG = load_config()
RES = ROOT / "results" / "16_deconv" / "prepare"


def deconv_samples() -> tuple[pd.DataFrame, pd.Index]:
    s, units = import_module("11_injury_axis").load_samples()
    uids = set().union(*[set(v) for v in units.values()]) | set(s.index[s.keep_for_pretrain])
    return s, pd.Index(sorted(uids), name="sample_uid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--references", default=",".join(DC["references"]))
    ap.add_argument("--skip-mixtures", action="store_true")
    args = ap.parse_args()
    set_global_seed(CFG["seed"])
    RES.mkdir(parents=True, exist_ok=True)
    genes = bulk_gene_union()
    summ, unm, drop = [], [], []
    for ds in args.references.split(","):
        for comp in sorted(set(DC["ref_compartment"].values())):
            info = build_reference(ds, comp, genes)
            summ.append(reference_summary(info))
            unm.append(info["unmapped"].rename("n_cells").reset_index().assign(dataset=ds, compartment=comp))
            drop.append(info["dropped_states"].rename("n_rows").reset_index().assign(dataset=ds, compartment=comp))
            print(f"[ref] {ds} {comp}: {len(info['rows'])} 行, {info['rows'].state.nunique()} 状态, "
                  f"{info['rows'].type.nunique()} 类型, {info['n_genes']} 基因", flush=True)
    for f, new in (("reference_states.tsv", pd.concat(summ)), ("reference_unmapped.tsv", pd.concat(unm)),
                   ("reference_dropped_states.tsv", pd.concat(drop))):
        p = RES / f
        if p.exists():                                   # 只重建部分参照时保留其余参照的行
            old = pd.read_csv(p, sep="\t")
            new = pd.concat([old[~old.dataset.isin(new.dataset.unique())], new])
        new.to_csv(p, sep="\t", index=False)
    if args.skip_mixtures:
        return

    s, uids = deconv_samples()
    man = []
    for (ser, comp), idx in mixture_groups(uids, s).items():
        p = write_mixture(ser, comp, idx, s)
        man.append({"series": ser, "compartment": comp, "reference_compartment": DC["ref_compartment"][comp],
                    "technology": s.loc[idx[0], "technology"], "n_samples": len(idx),
                    "file": str(p.relative_to(OUT))})
        print(f"[mix] {ser} {comp}: {len(idx)}", flush=True)
    M = pd.DataFrame(man)
    M.to_csv(RES / "mixture_manifest.tsv", sep="\t", index=False)
    pd.Series(uids).to_csv(RES / "deconv_samples.tsv", sep="\t", index=False)
    write_provenance("16_deconv/prepare", [ROOT / "data/processed/samples.tsv", ROOT / "data/processed/cohorts.tsv"],
                     [RES / f for f in ("reference_states.tsv", "reference_unmapped.tsv", "mixture_manifest.tsv",
                                        "deconv_samples.tsv")],
                     CFG["seed"], {"n_samples": len(uids), "n_batches": len(M), "out_dir": str(OUT)})


if __name__ == "__main__":
    main()
