## Validate expression-vs-activity Spearman correlations in R
## Compare with pre-computed Python results from cima_correlations.csv
##
## Levels tested:
##   donor_only  = bulk-like (all cells per donor pooled)
##   donor_l1    = donor × cell_type_l1
##   donor_l2    = donor × cell_type_l2
##   donor_l3    = donor × cell_type_l3
##   donor_l4    = donor × cell_type_l4
##
## Requirements: install.packages(c("anndata", "reticulate"))
## Make sure reticulate uses the secactpy conda env.

library(anndata)
library(Matrix)

base_dir <- "/data/parks34/projects/2secactpy/results/cross_sample_validation/cima"

# ---------- helper: read H5AD, return dense matrix + obs ---------
read_h5 <- function(path) {
  cat("  Loading", basename(path), "...")
  ad <- read_h5ad(path)
  mat <- as.matrix(ad$X)                # samples × features
  rownames(mat) <- rownames(ad$obs)
  colnames(mat) <- rownames(ad$var)
  cat(" done (", nrow(mat), "×", ncol(mat), ")\n")
  list(mat = mat, obs = ad$obs)
}

# ---------- helper: compute correlations for one level -----------
compute_level_cors <- function(expr, act, ref_level, ct_col = NULL) {
  # "all" = correlate across every row
  ref_all <- ref_level[ref_level$celltype == "all", ]
  r_all <- sapply(seq_len(nrow(ref_all)), function(i) {
    g <- ref_all$gene[i]; t <- ref_all$target[i]
    if (!(g %in% colnames(expr$mat)) || !(t %in% colnames(act$mat))) return(NA)
    cor(expr$mat[, g], act$mat[, t], method = "spearman")
  })
  comp_all <- data.frame(
    target = ref_all$target, gene = ref_all$gene, celltype = "all",
    rho_python = ref_all$spearman_rho, rho_R = r_all,
    diff = r_all - ref_all$spearman_rho
  )

  # Per-celltype (if ct_col provided)
  comp_ct <- NULL
  if (!is.null(ct_col)) {
    ref_ct <- ref_level[ref_level$celltype != "all", ]
    if (nrow(ref_ct) > 0) {
      r_ct <- sapply(seq_len(nrow(ref_ct)), function(i) {
        g  <- ref_ct$gene[i]; t <- ref_ct$target[i]; ct <- ref_ct$celltype[i]
        idx <- which(expr$obs[[ct_col]] == ct)
        if (length(idx) < 5) return(NA)
        if (!(g %in% colnames(expr$mat)) || !(t %in% colnames(act$mat))) return(NA)
        cor(expr$mat[idx, g], act$mat[idx, t], method = "spearman")
      })
      comp_ct <- data.frame(
        target = ref_ct$target, gene = ref_ct$gene, celltype = ref_ct$celltype,
        rho_python = ref_ct$spearman_rho, rho_R = r_ct,
        diff = r_ct - ref_ct$spearman_rho
      )
    }
  }

  list(all = comp_all, per_ct = comp_ct)
}

# ---------- Load reference CSV -----------------------------------
ref <- read.csv(file.path(dirname(base_dir), "correlations", "cima_correlations.csv"))

# ---------- 1. Donor-only (bulk) level ---------------------------
cat("\n=== Donor-only (bulk) level ===\n")
expr_bulk <- read_h5(file.path(base_dir, "cima_donor_only_pseudobulk.h5ad"))
act_bulk  <- read_h5(file.path(base_dir, "cima_donor_only_cytosig.h5ad"))

ref_bulk <- ref[ref$level == "donor_only" & ref$signature == "cytosig", ]
res_bulk <- compute_level_cors(expr_bulk, act_bulk, ref_bulk, ct_col = NULL)
cat("  max |diff| =", max(abs(res_bulk$all$diff), na.rm = TRUE), "\n")
print(head(res_bulk$all, 5))

# ---------- 2-5. Donor × cell_type_l1..l4 -----------------------
level_info <- list(
  list(level = "donor_l1", ct_col = "cell_type_l1"),
  list(level = "donor_l2", ct_col = "cell_type_l2"),
  list(level = "donor_l3", ct_col = "cell_type_l3"),
  list(level = "donor_l4", ct_col = "cell_type_l4")
)

res_levels <- list()   # store results for final plot
res_levels[["donor_only"]] <- res_bulk

for (info in level_info) {
  lvl    <- info$level
  ct_col <- info$ct_col
  cat("\n===", lvl, "===\n")

  expr_lx <- read_h5(file.path(base_dir, paste0("cima_", lvl, "_pseudobulk.h5ad")))
  act_lx  <- read_h5(file.path(base_dir, paste0("cima_", lvl, "_cytosig.h5ad")))

  ref_lx <- ref[ref$level == lvl & ref$signature == "cytosig", ]
  res_lx <- compute_level_cors(expr_lx, act_lx, ref_lx, ct_col = ct_col)

  cat("  all:        max |diff| =", max(abs(res_lx$all$diff), na.rm = TRUE), "\n")
  if (!is.null(res_lx$per_ct)) {
    cat("  per-celltype: max |diff| =", max(abs(res_lx$per_ct$diff), na.rm = TRUE), "\n")
  }
  print(head(res_lx$all, 3))

  res_levels[[lvl]] <- res_lx

  # Free memory
  rm(expr_lx, act_lx); gc(verbose = FALSE)
}

# ---------- Summary plot -----------------------------------------
cat("\n=== Summary plot ===\n")

# Gather all comparisons
plot_data <- do.call(rbind, lapply(names(res_levels), function(lvl) {
  res <- res_levels[[lvl]]
  rows <- res$all[, c("rho_python", "rho_R")]
  rows$level <- paste0(lvl, "_all")
  if (!is.null(res$per_ct)) {
    ct_rows <- res$per_ct[, c("rho_python", "rho_R")]
    ct_rows$level <- paste0(lvl, "_per_ct")
    rows <- rbind(rows, ct_rows)
  }
  rows
}))
plot_data$level <- factor(plot_data$level)

# Color palette
n_levels <- nlevels(plot_data$level)
pal <- rainbow(n_levels, s = 0.7, v = 0.85)

plot(plot_data$rho_python, plot_data$rho_R,
     pch = 16, cex = 0.3,
     col = pal[as.numeric(plot_data$level)],
     xlab = "Python (pre-computed)", ylab = "R (recomputed)",
     main = "Spearman rho: Python vs R (CytoSig, CIMA)")
abline(0, 1, lty = 2, col = "grey40")
legend("topleft",
       legend = levels(plot_data$level),
       col = pal, pch = 16, cex = 0.6, ncol = 2)
