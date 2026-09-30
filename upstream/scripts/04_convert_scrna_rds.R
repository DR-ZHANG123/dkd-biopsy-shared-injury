#!/usr/bin/env Rscript
# GSE131882 is published as a zUMIs dgecounts.rds; convert it to 10x-style mtx for scanpy.
suppressMessages(library(Matrix))

root <- normalizePath(file.path(dirname(sub("--file=", "",
        grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), ".."))
indir <- file.path(root, "data_raw", "scrna", "GSE131882", "raw")
outbase <- file.path(root, "data_raw", "scrna", "GSE131882", "mtx")

pick_matrix <- function(obj) {
  # zUMIs structure: obj$umicount$exon$all (or $inex$all)
  for (a in c("umicount", "readcount")) {
    if (!is.null(obj[[a]])) {
      for (b in c("exon", "inex", "intron")) {
        m <- obj[[a]][[b]][["all"]]
        if (!is.null(m)) return(list(mat = m, slot = paste(a, b, sep = "/")))
      }
    }
  }
  stop("no count matrix found in RDS")
}

for (f in list.files(indir, pattern = "\\.dgecounts\\.rds$", full.names = TRUE)) {
  sample <- sub("^GSM[0-9]+_", "", sub("\\.dgecounts\\.rds$", "", basename(f)))
  gsm <- sub("_.*$", "", basename(f))
  obj <- readRDS(f)
  got <- pick_matrix(obj)
  m <- got$mat
  cat(sprintf("[%s / %s] slot=%s  %d genes x %d barcodes\n",
              gsm, sample, got$slot, nrow(m), ncol(m)))

  od <- file.path(outbase, sample)
  dir.create(od, recursive = TRUE, showWarnings = FALSE)
  writeMM(as(m, "CsparseMatrix"), file.path(od, "matrix.mtx"))
  write.table(rownames(m), file.path(od, "features.tsv"),
              quote = FALSE, row.names = FALSE, col.names = FALSE)
  write.table(colnames(m), file.path(od, "barcodes.tsv"),
              quote = FALSE, row.names = FALSE, col.names = FALSE)
  system(paste("gzip -f", file.path(od, "matrix.mtx"),
               file.path(od, "features.tsv"), file.path(od, "barcodes.tsv")))
}
cat("done\n")
