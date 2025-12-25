# rm(list = ls())
# ==============================================================================
# Script: kdvine_sparsity.R
# Description: Evaluates the KDVine baseline on block-sparse Gaussian 
#              distributions. Models the dependence structure via nonparametric 
#              vine copulas and univariate kernel density marginals.
# Note: Prepared for anonymous submission.
# ==============================================================================

library(kdevine)
library(here)

# Load core modules
source(here("utils.R"))
source(here("dgp.R"))

# ------------------------------------------------------------------------------
# 1. Helper Functions
# ------------------------------------------------------------------------------

#' Transform Copula Samples to Original Data Space
#' Maps samples from the unit hypercube back to the original feature space 
#' using empirical quantile functions.
transform_to_original <- function(U_sim, x, marginal_fits) {
  n <- nrow(U_sim)
  d <- ncol(U_sim)
  X_sim <- matrix(0, n, d)
  
  for (j in 1:d) {
    # Invert through marginal quantiles
    X_sim[, j] <- quantile(x[, j], probs = U_sim[, j], type = 7)
  }
  return(X_sim)
}

# ------------------------------------------------------------------------------
# 2. Configuration & Experiment Setup
# ------------------------------------------------------------------------------

SAVE     <- TRUE
n        <- 1000            # Training sample size
p        <- 30              # Total dimensions
rho      <- 0.5             # Within-block correlation
N_BLOCKS <- c(3, 6, 10, 30) # Block sizes to evaluate
R        <- 30              # Repetitions

# Initialize storage
results_all <- vector("list", R)

# ------------------------------------------------------------------------------
# 3. Sparsity Experiment Loop
# ------------------------------------------------------------------------------

for (r_ in 1:R) {
  cat(sprintf("\n--- kdevine sparsity repetition %d / %d ---\n", r_, R))
  
  results_rep <- vector("list", length(N_BLOCKS))
  
  for (idx in seq_along(N_BLOCKS)) {
    n_blocks <- N_BLOCKS[idx]
    
    # 3.1. Data Generation
    x <- block_gaussian(n, n_blocks = n_blocks, p = p, rho = rho)
    
    # 3.2. Fit Kernel Marginals
    marginal_fits <- lapply(1:p, function(j) {
      kde1d(x[, j],
            xmin = min(x[, j]) - 0.1 * sd(x[, j]),
            xmax = max(x[, j]) + 0.1 * sd(x[, j]))
    })
    
    # 3.3. Fit Vine Copula
    u   <- VineCopula::pobs(x, ties = "average")
    fit <- kdevinecop(u)
    
    # 3.4. Sampling & Inverse Transform
    u_sampling <- rkdevinecop(1000, fit) # Sample size limited for performance
    x_sampling <- transform_to_original(u_sampling, x, marginal_fits)
    
    # 3.5. Accuracy Evaluation
    x_eval <- block_gaussian(10000, n_blocks = n_blocks, p = p, rho = rho)
    err    <- cramer_wold_distance(x_sampling, x_eval)
    
    cat(sprintf("kdevine | Error: %.6f | Block Size: %d\n", err, n_blocks))
    
    # 3.6. Store result
    results_rep[[idx]] <- data.frame(
      n_blocks   = n_blocks,
      err        = err,
      repetition = r_
    )
  }
  
  # Aggregate results for this repetition
  results_all[[r_]] <- do.call(rbind, results_rep)
}

# ------------------------------------------------------------------------------
# 4. Save Results
# ------------------------------------------------------------------------------

if (SAVE) {
  res_path <- here("simulation_experiments", "simulation_results", "kdvine_sparsity.rds")
  dir.create(dirname(res_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(results_all, res_path)
  cat("\nResults successfully saved to:", res_path, "\n")
}

