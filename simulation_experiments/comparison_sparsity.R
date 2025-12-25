# rm(list = ls())
library(here)
library(xtable)
library(dplyr)
library(tidyr)

# Load results
results_kdvine <- readRDS(here::here("simulation_experiments/simulation_results/kdvine_sparsity.rds"))
results_kdvine <- do.call(rbind, results_kdvine)
results_band <- readRDS(here::here("simulation_experiments/simulation_results/band_sparsity.rds"))
results_band <- do.call(rbind, results_band)
results_mcluster <- readRDS(here::here("simulation_experiments/simulation_results/mcluster_sparsity.rds"))
results_mcluster <- do.call(rbind, results_mcluster)
results_nflow <- read.csv(here::here("simulation_experiments/simulation_results/nflow_sparsity.csv"))



# Data processing function
make_row <- function(model_name, results_df) {
  
  # === 2. Aggregate err and err_diag (now with min, mean, max) ===
  results_df <- aggregate(
    cbind(err) ~ n_blocks,
    data = results_df,
    FUN = function(x) c(min = min(x), mean = mean(x), max = max(x), sd = sd(x))
  )
  
  # === 3. Convert nested columns into separate numeric columns ===
  results_df <- do.call(data.frame, results_df)
  
  # === 4. Rename columns clearly ===
  names(results_df)[names(results_df) == "err.min"]        <- "err_min"
  names(results_df)[names(results_df) == "err.mean"]       <- "err_mean"
  names(results_df)[names(results_df) == "err.max"]        <- "err_max"
  names(results_df)[names(results_df) == "err.sd"]        <- "err_sd"
  
  
  
  # 5. Ensure sorted by n_blocks
  results_df <- results_df[order(results_df$n_blocks), ]
  
  # 6, Create "mean (sd)" strings
  cells <- sprintf("%.3f (%.3f)", results_df$err_mean, results_df$err_sd)
  
  # 7. Collapse into one row
  row <- paste(c(model_name, cells), collapse = " & ")
  
  return(row)
}


cat(make_row("BAND", results_band), "\\\\")
cat(make_row("mcluster", results_mcluster), '\\\\')
cat(make_row("kdvine", results_kdvine), '\\\\')
cat(make_row("nflow", results_nflow), '\\\\')
