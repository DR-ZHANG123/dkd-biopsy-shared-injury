#!/usr/bin/env python
"""RMA in Python: background adjustment, quantile normalisation, median polish.

R is used only to parse the CEL files and locate PM probes; affy's rma() and
bg.correct() are unavailable here because the pthread backend of preprocessCore
fails on high-core-count machines. The three numerical steps are reimplemented
following affy's own definitions, and 03c validates the result against the
deposited processing."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
PM_DIR = ROOT / "data_processed" / "bulk" / "pm_export"
OUT = ROOT / "data_processed" / "bulk"


def epanechnikov_mode(x: np.ndarray, n_pts: int = 1 << 14) -> float:
    """Mode of an Epanechnikov kernel density estimate, as affy uses for the background mode."""
    x = x[np.isfinite(x)]
    n = x.size
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    lo = min(sd, iqr / 1.349) if iqr > 0 else sd
    if lo <= 0:
        lo = abs(x[0]) if x[0] != 0 else 1.0
    bw = 0.9 * lo * n ** (-0.2)
    h = bw * np.sqrt(5.0)
    cut = 3 * bw
    grid = np.linspace(x.min() - cut, x.max() + cut, n_pts)
    counts, edges = np.histogram(x, bins=n_pts, range=(grid[0], grid[-1]))
    centers = 0.5 * (edges[:-1] + edges[1:])
    delta = centers[1] - centers[0]
    half = int(np.ceil(h / delta))
    off = np.arange(-half, half + 1) * delta
    kern = np.maximum(0.0, 0.75 * (1 - (off / h) ** 2) / h)
    kern /= kern.sum()
    dens = np.convolve(counts.astype(float), kern, mode="same")
    return float(centers[int(np.argmax(dens))])


def bg_parameters(pm: np.ndarray) -> tuple[float, float, float]:
    """Estimate the exponential-signal and normal-noise parameters of the RMA background model."""
    mu = epanechnikov_mode(pm)
    bg = pm[pm < mu] - mu
    sigma = np.sqrt((bg ** 2).sum() / (bg.size - 1)) * np.sqrt(2.0)
    sig = pm[pm > mu] - mu
    alpha = 1.0 / epanechnikov_mode(sig)
    return alpha, mu, sigma


def bg_adjust(pm: np.ndarray) -> np.ndarray:
    """Apply the RMA background correction to one array."""
    out = np.empty_like(pm)
    for j in range(pm.shape[1]):
        col = pm[:, j]
        alpha, mu, sigma = bg_parameters(col)
        a = col - mu - alpha * sigma ** 2
        out[:, j] = a + sigma * norm.pdf(a / sigma) / norm.cdf(a / sigma)
    return out


def quantile_normalize(mat: np.ndarray) -> np.ndarray:
    """Quantile-normalise columns to their common mean distribution."""
    order = np.argsort(mat, axis=0, kind="mergesort")
    sorted_mat = np.take_along_axis(mat, order, axis=0)
    target = sorted_mat.mean(axis=1)
    out = np.empty_like(mat)
    rows = np.arange(mat.shape[0])
    for j in range(mat.shape[1]):
        out[order[:, j], j] = target
    del rows
    return out


def median_polish(block: np.ndarray, max_iter: int = 10,
                  eps: float = 1e-4) -> np.ndarray:
    """Median polish of a probe-by-sample matrix, returning the sample effects."""
    z = block.copy()
    nr, nc = z.shape
    row_eff = np.zeros(nr)
    col_eff = np.zeros(nc)
    overall = 0.0
    old_sum = 0.0
    for _ in range(max_iter):
        rmed = np.median(z, axis=1)
        z -= rmed[:, None]
        row_eff += rmed
        delta = np.median(col_eff)
        col_eff -= delta
        overall += delta

        cmed = np.median(z, axis=0)
        z -= cmed[None, :]
        col_eff += cmed
        delta = np.median(row_eff)
        row_eff -= delta
        overall += delta

        new_sum = np.abs(z).sum()
        if old_sum and abs(new_sum - old_sum) < eps * new_sum:
            break
        old_sum = new_sum
    return overall + col_eff


def rma_one(gse: str) -> pd.DataFrame:
    shape = [int(v) for v in
             (PM_DIR / f"{gse}_shape.txt").read_text().split()]
    pm = np.fromfile(PM_DIR / f"{gse}_pm.bin", dtype="<f8").reshape(
        shape, order="F")
    with gzip.open(PM_DIR / f"{gse}_probesets.txt.gz", "rt") as fh:
        psets = np.array([ln.strip() for ln in fh])
    samples = (PM_DIR / f"{gse}_samples.txt").read_text().split()
    assert pm.shape == (len(psets), len(samples)), (pm.shape, len(psets), len(samples))
    print(f"[{gse}] PM {pm.shape}")

    pm = bg_adjust(pm)
    pm = quantile_normalize(pm)
    logpm = np.log2(np.maximum(pm, 1e-8))

    order = np.argsort(psets, kind="mergesort")
    psets_s, logpm_s = psets[order], logpm[order]
    uniq, starts = np.unique(psets_s, return_index=True)
    bounds = np.append(starts, len(psets_s))

    expr = np.empty((uniq.size, logpm.shape[1]))
    for i in range(uniq.size):
        expr[i] = median_polish(logpm_s[bounds[i]:bounds[i + 1]])
    df = pd.DataFrame(expr, index=uniq, columns=samples)
    df.index.name = "probe"
    print(f"[{gse}] summarised -> {df.shape[0]} probesets x {df.shape[1]} arrays")
    return df


def main() -> None:
    for gse in ("GSE30528", "GSE30529"):
        df = rma_one(gse)
        out = OUT / f"{gse}_rma_probe.tsv.gz"
        df.to_csv(out, sep="\t")
        print(f"[{gse}] wrote {out}")


if __name__ == "__main__":
    main()
