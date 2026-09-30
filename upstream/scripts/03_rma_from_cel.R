#!/usr/bin/env Rscript
# The series matrices for GSE30528 / GSE30529 are probe-centred log ratios (row means
# near zero), so the within-sample ordering is destroyed and RMA must be redone from CEL.
#
# The pthread backend of preprocessCore / affy returns EINVAL on high-core-count
# machines, so rma() and bg.correct() are unavailable (recompiling did not help).
# affy is therefore used only to parse the CEL files and locate PM probes (pure R/C,
# no threads); the three numerical steps of RMA (background correction, quantile
# normalisation, median polish) are implemented in scripts/03b_rma_summarize.py
suppressMessages({library(affy); library(hgu133a2cdf)})

root <- normalizePath(file.path(dirname(sub("--file=", "",
        grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), ".."))
outdir <- file.path(root, "data_processed", "bulk", "pm_export")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

for (gse in c("GSE30528", "GSE30529")) {
  celdir <- file.path(root, "data_raw", "bulk", gse, "cel")
  files <- sort(list.files(celdir, pattern = "\\.CEL\\.gz$", full.names = TRUE))
  stopifnot(length(files) > 0)

  ab <- ReadAffy(filenames = files)
  pmmat <- pm(ab)
  psets <- probeNames(ab)
  samples <- sub("_.*$", "", basename(files))
  cat(sprintf("[%s] %d PM probes x %d arrays, %d probesets\n",
              gse, nrow(pmmat), ncol(pmmat), length(unique(psets))))

  # float64 little-endian binary, column-major (probes x arrays)
  con <- file(file.path(outdir, paste0(gse, "_pm.bin")), "wb")
  writeBin(as.vector(pmmat), con, size = 8)
  close(con)
  writeLines(psets, gzfile(file.path(outdir, paste0(gse, "_probesets.txt.gz"))))
  writeLines(samples, file.path(outdir, paste0(gse, "_samples.txt")))
  writeLines(as.character(c(nrow(pmmat), ncol(pmmat))),
             file.path(outdir, paste0(gse, "_shape.txt")))
}
cat("PM export done\n")
