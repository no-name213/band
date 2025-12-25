# rm(list = ls())
# ==============================================================================
# Script: kdvine_monte_carlo.R
# Description: Benchmark evaluation using nonparametric vine copula density 
#              estimation (kdevine). Models multivariate distributions via 
#              kernel marginals and nonparametric vine copulas.
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
#' Uses the empirical quantile function of the original data to invert 
#' the probability integral transform.
transform_to_original <- function(U_sim, x, marginal_fits) {
  n <- nrow(U_sim)
  d <- ncol(U_sim)
  X_sim <- matrix(0, n, d)
  
  for (j in 1:d) {
    # Invert through marginal quantiles (type 7 is R default)
    X_sim[, j] <- quantile(x[, j], probs = U_sim[, j], type = 7)
  }
  return(X_sim)
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
  cat(sprintf("\n--- kdevine repetition %d / %d ---\n", r_, R))
  
  idx <- 1
  results_rep <- vector("list", NUM_CFG)
  
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
          
          # 3.2. Fit Kernel Marginals
          marginal_fits <- lapply(1:p, function(j) {
            kde1d(x[, j],
                  xmin = min(x[, j]) - 0.1 * sd(x[, j]),
                  xmax = max(x[, j]) + 0.1 * sd(x[, j]))
          })
          
          # 3.3. Fit Vine Copula (Nonparametric)
          u   <- VineCopula::pobs(x, ties = "average")
          fit <- kdevinecop(u)
          
          # 3.4. Sampling & Inverse Transform
          # kdvine sampling is slow
          u_sampling <- rkdevinecop(1000, fit)
          x_sampling <- transform_to_original(u_sampling, x, marginal_fits)
          
          # 3.5. Accuracy Evaluation
          if (model == 'unit') {
            x_eval <- sample_bimodal_uniform(10000, p = p, d = d)
          } else {
            x_eval <- rmixnorm(10000, p = p, d = d)
          }
          
          err <- cramer_wold_distance(x_sampling, x_eval)
          
          cat(sprintf("kdevine | Error: %.6f | Model: %s | n: %d | p: %d | d: %d\n",
                      err, model, n, p, d))
          
          # 3.6. Store result configuration
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
  results_all[[r_]] <- do.call(rbind, results_rep)
}

# ------------------------------------------------------------------------------
# 4. Save Results
# ------------------------------------------------------------------------------

if (SAVE) {
  res_path <- here("simulation_experiments", "simulation_results", "kdvine_monte_carlo.rds")
  dir.create(dirname(res_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(results_all, res_path)
  cat("\nResults successfully saved to:", res_path, "\n")
}


