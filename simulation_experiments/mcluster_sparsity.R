# rm(list = ls())
# ==============================================================================
# Script: mcluster_sparsity.R
# Description: Evaluates the Mclust (GMM) baseline on block-sparse Gaussian 
#              distributions. Measures the performance of parametric mixture 
#              models in capturing structured covariance.
# Note: Prepared for anonymous submission.
# ==============================================================================

library(mclust)
library(MASS)
library(here)

# Load core modules
source(here("utils.R"))
source(here("dgp.R"))

# ------------------------------------------------------------------------------
# 1. Helper Functions
# ------------------------------------------------------------------------------

#' Simulate from a Fitted Gaussian Mixture
#' @param params List containing 'pro' (weights), 'mean', and 'sigma'.
#' @param N Number of samples to generate.
simulate_mixture <- function(params, N) {
  k <- length(params$pro)
  p <- nrow(params$mean)
  z <- sample(1:k, size = N, replace = TRUE, prob = params$pro)
  X <- matrix(0, N, p)
  for (i in 1:k) {
    idx <- which(z == i)
    if (length(idx) > 0) {
      X[idx, ] <- MASS::mvrnorm(
        length(idx),
        mu = params$mean[, i],
        Sigma = params$sigma[, , i]
      )
    }
  }
  return(X)
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
  cat(sprintf("\n--- Mclust sparsity repetition %d / %d ---\n", r_, R))
  
  results_rep <- vector("list", length(N_BLOCKS))
  
  for (idx in seq_along(N_BLOCKS)) {
    n_blocks <- N_BLOCKS[idx]
    
    # 3.1. Data Generation
    x <- block_gaussian(n, n_blocks = n_blocks, p = p, rho = rho)
    
    # 3.2. Fit Mclust (G = 2 as per experimental setup)
    gfit <- Mclust(x, G = 2, verbose = FALSE)
    
    # Extract parameters for sampling
    params <- list(
      pro   = gfit$parameters$pro,
      mean  = gfit$parameters$mean,
      sigma = gfit$parameters$variance$sigma
    )
    
    # 3.3. Simulate from Fitted Model
    x_sampling <- simulate_mixture(params, N = 3000)
    
    # 3.4. Accuracy Evaluation (Cramér–Wold Distance)
    x_eval <- block_gaussian(10000, n_blocks = n_blocks, p = p, rho = rho)
    err    <- cramer_wold_distance(x_sampling, x_eval)
    
    cat(sprintf("Mclust | Error: %.6f | Block Size: %d\n", err, n_blocks))
    
    # 3.5. Store result
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
  res_path <- here("simulation_experiments", "simulation_results", "mcluster_sparsity.rds")
  dir.create(dirname(res_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(results_all, res_path)
  cat("\nResults successfully saved to:", res_path, "\n")
}


