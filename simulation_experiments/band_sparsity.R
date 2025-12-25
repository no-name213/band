# rm(list = ls())
# ==============================================================================
# Script: band_sparsity.R
# Description: Evaluates the BAND framework's performance on block-sparse 
#              Gaussian distributions across varying block sizes.
# Note: Prepared for anonymous submission.
# ==============================================================================

library(here)

# Load core modules from the project root
source(here("utils.R"))
source(here("train.R"))
source(here("dgp.R"))

# ------------------------------------------------------------------------------
# 1. Configuration & Global Parameters
# ------------------------------------------------------------------------------

SAVE     <- TRUE
n        <- 1000            # Training samples
p        <- 30              # Total dimensions
rho      <- 0.5             # Correlation within blocks
N_BLOCKS <- c(3, 6, 10, 30) # Variables per block
R        <- 30              # Number of experiment repetitions

# Initialize storage per repetition
results_all <- vector("list", R)

# ------------------------------------------------------------------------------
# 2. Experiment Loop
# ------------------------------------------------------------------------------

for (r_ in 1:R) {
  cat(sprintf("\n--- Repetition %d / %d ---\n", r_, R))
  
  idx <- 1
  results_rep <- vector("list", length(N_BLOCKS))
  
  for (n_blocks in N_BLOCKS) {
    # 2.1. Data Generation
    # Generates a p-dimensional Gaussian with block-diagonal covariance
    x <- block_gaussian(n, n_blocks = n_blocks, p = p, rho = rho)
    
    # 2.2. Model Training
    # need_presampling = FALSE is used as we only require the sampler for CW distance
    band_sampler <- band(x, need_presampling = FALSE)
    
    # 2.3. Sampling & Evaluation
    # Generate synthetic data for accuracy assessment
    x_sampling <- band_sampler(10000)
    
    # Generate ground truth test data
    x_eval <- block_gaussian(10000, n_blocks = n_blocks, p = p, rho = rho)
    
    # Compute accuracy metric (Cramér–Wold Distance)
    err <- cramer_wold_distance(x_sampling, x_eval)
    
    cat(sprintf("BAND | Block Size: %2d | rho: %.2f | CW Distance: %.6f\n", 
                n_blocks, rho, err))
    
    # 2.4. Store per-repetition result configuration
    results_rep[[idx]] <- data.frame(
      err        = err,
      n_blocks   = n_blocks,
      repetition = r_
    )
    
    idx <- idx + 1
  }
  
  # Aggregate results for this repetition
  results_all[[r_]] <- do.call(rbind, results_rep)
}

# ------------------------------------------------------------------------------
# 3. Save Results
# ------------------------------------------------------------------------------

if (SAVE) {
  res_path <- here("simulation_experiments", "simulation_results", "band_sparsity.rds")
  
  # Ensure the directory exists before saving
  dir.create(dirname(res_path), showWarnings = FALSE, recursive = TRUE)
  
  saveRDS(results_all, res_path)
  cat("\nResults successfully saved to:", res_path, "\n")
}
