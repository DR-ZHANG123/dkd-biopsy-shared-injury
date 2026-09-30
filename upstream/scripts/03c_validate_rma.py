#!/usr/bin/env python
"""Validate the Python RMA against the deposited probe-centred matrices.

Agreement is assessed after re-centring, because the deposited values are
probe-centred log ratios and cannot be compared on an absolute scale."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_series_matrix(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
        df = pd.read_csv(fh, sep="\t", index_col=0, comment="!",
                         na_values=["", "NA", "null"])
    df.index = df.index.astype(str).str.strip('"')
    df.columns = [c.strip('"') for c in df.columns]
    return df.dropna(how="all")


def main() -> None:
    rows = []
    for gse in ("GSE30528", "GSE30529"):
        geo = read_series_matrix(
            ROOT / "data_raw" / "bulk" / gse / f"{gse}_series_matrix.txt.gz")
        mine = pd.read_csv(
            ROOT / "data_processed" / "bulk" / f"{gse}_rma_probe.tsv.gz",
            sep="\t", index_col=0)

        probes = geo.index.intersection(mine.index)
        samples = [c for c in geo.columns if c in mine.columns]
        g = geo.loc[probes, samples].astype(float)
        m = mine.loc[probes, samples].astype(float)
        gc = g.sub(g.mean(axis=1), axis=0)
        mc = m.sub(m.mean(axis=1), axis=0)

        keep = mc.std(axis=1) > 0
        per_probe = gc[keep].corrwith(mc[keep], axis=1)
        per_sample = gc.corrwith(mc, axis=0)

        rows.append({
            "gse": gse, "n_probes": int(keep.sum()), "n_samples": len(samples),
            "probe_r_median": round(float(per_probe.median()), 4),
            "probe_r_q25": round(float(per_probe.quantile(.25)), 4),
            "probe_frac_r_gt_0.8": round(float((per_probe > .8).mean()), 4),
            "sample_r_median": round(float(per_sample.median()), 4),
            "sample_r_min": round(float(per_sample.min()), 4),
        })
        print(rows[-1])

        rank_r = [float(pd.Series(m[s]).corr(pd.Series(g[s]), method="spearman"))
                  for s in samples]
        print(f"  [{gse}] within-sample Spearman(own absolute values, deposited centred values) "
              f"median={np.median(rank_r):.3f} "
              f"- a low value means the deposited matrix cannot be used for within-sample ranking")

    out = ROOT / "results" / "tables" / "T00_rma_validation.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
