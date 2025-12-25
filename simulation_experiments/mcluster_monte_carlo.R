# rm(list = ls())
# ==============================================================================
# Script: mcluster_monte_carlo.R
# Description: Parametric benchmark using Gaussian Mixture Models (Mclust).
#              Evaluates the ability of GMMs to approximate target distributions
#              under varying dimensions, sample sizes, and shifts.
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

SAVE  <- TRUE
P     <- c(2, 10, 30)   # Dimensions
N     <- c(100, 1000)   # Sample sizes
D     <- c(0, 2, 4)     # Mean shifts
MODEL <- c('unit', 'gaussian')
R     <- 30             # Repetitions

# Initialize storage
NUM_CFG <- length(P) * length(N) * length(D) * length(MODEL)
results_all <- vector("list", R)

# ------------------------------------------------------------------------------
# 3. Monte Carlo Loop
# ------------------------------------------------------------------------------

for (r_ in 1:R) {
  cat(sprintf("\n--- Mclust repetition %d / %d ---\n", r_, R))
  
  idx <- 1
  res_rep <- vector("list", NUM_CFG)
  
  for (p in P) {
    for (n in N) {
      for (d in D) {
        for (model in MODEL) {
          
          # 3.1. Data Generation
          if (model == 'unit') {
            x <- sample_bimodal_uniform(n, p = p, d = d)
          } else {
            x <- rmixnorm(n, p = p, d = d)
          }
          
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
          
          # 3.4. Accuracy Evaluation
          if (model == 'unit') {
            x_eval <- sample_bimodal_uniform(10000, p = p, d = d)
          } else {
            x_eval <- rmixnorm(10000, p = p, d = d)
          }
          
          err <- cramer_wold_distance(x_sampling, x_eval)
          
          cat(sprintf("Mclust | Error: %.6f | Model: %s | n: %d | p: %d | d: %d\n",
                      err, model, n, p, d))
          
          # 3.5. Store result
          res_rep[[idx]] <- data.frame(
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
  results_all[[r_]] <- do.call(rbind, res_rep)
}

# ------------------------------------------------------------------------------
# 4. Save Results
# ------------------------------------------------------------------------------

if (SAVE) {
  res_path <- here("simulation_experiments", "simulation_results", "mcluster_monte_carlo.rds")
  dir.create(dirname(res_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(results_all, res_path)
  cat("\nResults successfully saved to:", res_path, "\n")
}

