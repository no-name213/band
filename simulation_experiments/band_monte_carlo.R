# rm(list = ls())
# ==============================================================================
# Script: band_monte_carlo.R
# Description: Evaluates the BAND framework's performance in approximating 
#              complex multivariate distributions via Monte Carlo simulations.
#              Tests various dimensions, sample sizes, and mean shifts.
# Note: Prepared for anonymous submission.
# ==============================================================================

library(here)

# Load core modules from the project root
source(here("utils.R"))
source(here("train.R"))
source(here("sampling.R"))
source(here("dgp.R"))

# ------------------------------------------------------------------------------
# 1. Configuration & Global Parameters
# ------------------------------------------------------------------------------

SAVE  <- TRUE
P     <- c(2, 10, 30)           # Dimensions
N     <- c(100, 1000)           # Training sample sizes
D     <- c(0, 2, 4)             # Mean shifts (bimodality distance)
MODEL <- c('unit', 'gaussian')  # Distribution types
R     <- 30                     # Number of experiment repetitions

# Initialize storage per repetition
NUM_CFG <- length(P) * length(N) * length(D) * length(MODEL)
results_all <- vector("list", R)

# ------------------------------------------------------------------------------
# 2. Monte Carlo Loop
# ------------------------------------------------------------------------------

for (r_ in 1:R) {
  cat(sprintf("\n--- BAND Repetition %d / %d ---\n", r_, R))
  
  idx <- 1
  results_rep <- vector("list", NUM_CFG)
  
  for (p in P) {
    for (n in N) {
      for (d in D) {
        for (model in MODEL) {
          
          # 2.1. Data Generation
          if (model == 'unit') {
            x <- sample_bimodal_uniform(n, p = p, d = d)
          } else {
            x <- rmixnorm(n, p = p, d = d)
          }
          
          # 2.2. Model Training
          # need_presampling = FALSE as we only require the sampler for CW distance
          band_sampler <- band(x, need_presampling = FALSE)
          
          # 2.3. Sampling & Evaluation
          # Generate synthetic data
          x_sampling <- band_sampler(10000)
          
          # Generate ground truth test data
          if (model == 'unit') {
            x_eval <- sample_bimodal_uniform(10000, p = p, d = d)
          } else {
            x_eval <- rmixnorm(10000, p = p, d = d)
          }
          
          # Compute accuracy metric (Cramér–Wold Distance)
          err <- cramer_wold_distance(x_sampling, x_eval)
          
          cat(sprintf("BAND | Error: %.6f | Model: %s | n: %d | p: %d | d: %d\n",
                      err, model, n, p, d))
          
          # 2.4. Store per-repetition result configuration
          results_rep[[idx]] <- data.frame(
            model      = model,
            err        = err,
            p          = p,
            n          = n,
            d          = d,
            repetition = r_
          )
          
          idx <- idx + 1
        }
      }
    }
  }
  # Aggregate results for this repetition
  results_all[[r_]] <- do.call(rbind, results_rep)
}

# ------------------------------------------------------------------------------
# 3. Save Results
# ------------------------------------------------------------------------------

if (SAVE) {
  res_path <- here("simulation_experiments", "simulation_results", "band_monte_carlo.rds")
  
  # Ensure the directory exists before saving
  dir.create(dirname(res_path), showWarnings = FALSE, recursive = TRUE)
  
  saveRDS(results_all, res_path)
  cat("\nResults successfully saved to:", res_path, "\n")
}
