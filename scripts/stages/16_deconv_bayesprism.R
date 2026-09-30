# Stage 16b：对一个 (series, 区室) 混合物运行 BayesPrism（由 16_deconv_run.py 调用，参数全部来自 config/run.yaml）。
#
# 参照：16_deconv_prepare.py 写出的 KPMP 供体 × 状态 pseudobulk（counts_triplet.parquet / rows.tsv / genes.txt）。
# 1) cleanup.genes（核糖体 / 线粒体 / 性染色体 / MALAT1）+ 只保留蛋白编码基因；
# 2) 标志基因：get.exp.stat（细胞类型两两比较，状态信息用于分层）+ select.marker；结果按参照缓存 markers.txt；
# 3) new.prism（key = NULL：无肿瘤细胞，参照更新为跨样本共享的逐基因平台偏移）→ run.prism；
# 4) 输出 theta_type（final）、theta_state（first，状态层面）、theta_cv、Z_type（初始 Gibbs 的细胞类型特异表达，逐类型）。
#
# 用法：Rscript 16_deconv_bayesprism.R ref=<dir> mix=<tsv.gz> out=<dir> n_cores=64 seed=1 chain_length=1000 burn_in=500
#       thinning=2 outlier_cut=0.01 outlier_fraction=0.1 pval_max=0.01 lfc_min=0.1 cell_count_cutoff=3
#       pseudo_count=0.1 gene_groups=Rb,Mrp,... protein_coding=TRUE
suppressPackageStartupMessages({
  library(BayesPrism)
  library(Matrix)
  library(arrow)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
kv <- setNames(sub("^[^=]*=", "", args), sub("=.*$", "", args))
num <- function(k) as.numeric(kv[[k]])
ref_dir <- kv[["ref"]]; mix_file <- kv[["mix"]]; out_dir <- kv[["out"]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
n_cores <- as.integer(kv[["n_cores"]])
gene_groups <- strsplit(kv[["gene_groups"]], ",")[[1]]

load_reference <- function(d) {
  tr <- as.data.frame(read_parquet(file.path(d, "counts_triplet.parquet")))
  rows <- fread(file.path(d, "rows.tsv"), sep = "\t", quote = "")
  genes <- readLines(file.path(d, "genes.txt"))
  X <- sparseMatrix(i = tr$i, j = tr$j, x = tr$x, dims = c(nrow(rows), length(genes)))
  X <- as.matrix(X)
  rownames(X) <- rows$key; colnames(X) <- genes
  list(X = X, rows = rows)
}

clean_reference <- function(X) {
  X <- cleanup.genes(input = X, input.type = "count.matrix", species = "hs", gene.group = gene_groups, exp.cells = 1)
  if (as.logical(kv[["protein_coding"]])) X <- select.gene.type(X, gene.type = "protein_coding")
  X
}

markers_for <- function(ref, X) {
  f <- file.path(ref_dir, "markers.txt")
  if (file.exists(f)) return(readLines(f))
  stat <- get.exp.stat(sc.dat = X, cell.type.labels = ref$rows$type, cell.state.labels = ref$rows$state,
                       pseudo.count = num("pseudo_count"), cell.count.cutoff = num("cell_count_cutoff"),
                       n.cores = n_cores)
  Xm <- select.marker(sc.dat = X, stat = stat, pval.max = num("pval_max"), lfc.min = num("lfc_min"))
  mk <- colnames(Xm)
  writeLines(mk, f)
  saveRDS(stat, file.path(ref_dir, "exp_stat.rds"))
  mk
}

write_mat <- function(M, file, id = "sample_uid") {
  df <- data.frame(id = rownames(M), M, check.names = FALSE)
  names(df)[1] <- id
  write_parquet(df, file)
}

ref <- load_reference(ref_dir)
Xc <- clean_reference(ref$X)
mk <- markers_for(ref, Xc)
if (mix_file == "NONE") {                      # 仅计算并缓存标志基因（并行运行前每个参照先做一次）
  cat(sprintf("[bp] markers %s: %d\n", ref_dir, length(mk)))
  quit(save = "no", status = 0)
}
mix <- fread(mix_file, sep = "\t")
genes_mix <- mix$gene
M <- t(as.matrix(mix[, -1, with = FALSE])); colnames(M) <- genes_mix
use <- intersect(mk, genes_mix)
cat(sprintf("[bp] %s: %d samples, markers %d, used %d\n", basename(mix_file), nrow(M), length(mk), length(use)))
prism <- new.prism(reference = Xc[, use], mixture = M[, use, drop = FALSE], input.type = "count.matrix",
                   cell.type.labels = ref$rows$type, cell.state.labels = ref$rows$state, key = NULL,
                   outlier.cut = num("outlier_cut"), outlier.fraction = num("outlier_fraction"))
gc_ctrl <- list(chain.length = num("chain_length"), burn.in = num("burn_in"), thinning = num("thinning"),
                seed = as.integer(kv[["seed"]]), n.cores = n_cores)
bp <- run.prism(prism = prism, n.cores = n_cores, gibbs.control = gc_ctrl)

theta_t <- get.fraction(bp = bp, which.theta = "final", state.or.type = "type")
theta_t0 <- get.fraction(bp = bp, which.theta = "first", state.or.type = "type")
theta_s <- get.fraction(bp = bp, which.theta = "first", state.or.type = "state")
write_mat(theta_t, file.path(out_dir, "theta_type.parquet"))
write_mat(theta_t0, file.path(out_dir, "theta_type_first.parquet"))
write_mat(theta_s, file.path(out_dir, "theta_state.parquet"))
cv <- bp@posterior.theta_f@theta.cv
write_mat(cv, file.path(out_dir, "theta_cv.parquet"))
for (ct in colnames(theta_t)) {
  Z <- get.exp(bp = bp, state.or.type = "type", cell.name = ct)
  write_mat(Z, file.path(out_dir, sprintf("Z_%s.parquet", ct)))
}
writeLines(colnames(Z), file.path(out_dir, "genes_used.txt"))
writeLines("ok", file.path(out_dir, "DONE"))
