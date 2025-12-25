# rm(list = ls())
library(here)
library(xtable)
library(dplyr)
library(tidyr)

# Load results
results_kdvine <- readRDS(here::here("simulation_experiments/simulation_results/kdvine_monte_carlo.rds"))
results_kdvine <- do.call(rbind, results_kdvine)
results_band <- readRDS(here::here("simulation_experiments/simulation_results/band_monte_carlo.rds"))
results_band <- do.call(rbind, results_band)
results_mcluster <- readRDS(here::here("simulation_experiments/simulation_results/mcluster_monte_carlo.rds"))
results_mcluster <- do.call(rbind, results_mcluster)
results_nflow <- read.csv(here::here("simulation_experiments/simulation_results/nflow_monte_carlo.csv"))


# Data processing function
make_latex_table_compare <- function(df, method_name,
                                     model1 = "unit", model2 = "gaussian") {
  
  ### ------------------ FINAL SUMMARY (min/mean/max) ------------------ ###
  
  # Aggregate both err, now grouped by model as well
  df <- aggregate(
    cbind(err) ~ model + p + n + d,
    data = df,
    FUN = function(x) c(min = min(x), mean = mean(x), max = max(x), sd = sd(x))
  )
  
  # Convert nested columns into separate data frame columns
  df <- do.call(data.frame, df)
  
  # Rename columns for clarity
  names(df)[names(df) == "err.min"]       <- "err_min"
  names(df)[names(df) == "err.mean"]      <- "err_mean"
  names(df)[names(df) == "err.max"]       <- "err_max"
  names(df)[names(df) == "err.sd"]       <- "err_sd"
  
  
  # Format display column
  df <- df %>%
    mutate(display = paste0(sprintf("%.3f", round(err_mean,3)),
                            " (", sprintf("%.2f", round(err_sd,2)), ")"))
  
  # Separate models
  df1 <- df %>% filter(model == model1) %>% arrange(p, n, d) %>%
    select(p, n, d, display) %>%
    pivot_wider(names_from = d, values_from = display, names_prefix = "d=")
  
  df2 <- df %>% filter(model == model2) %>% arrange(p, n, d) %>%
    select(p, n, d, display) %>%
    pivot_wider(names_from = d, values_from = display, names_prefix = "d=")
  
  # Merge by p/n
  df_join <- inner_join(df1, df2, by = c("p","n"))
  
  # Build LaTeX rows
  rows <- apply(df_join, 1, function(r) {
    paste0(r["p"], "/", r["n"], " & ",
           r["d=0.x"], " & ", r["d=2.x"], " & ", r["d=4.x"], " & ",
           r["d=0.y"], " & ", r["d=2.y"], " & ", r["d=4.y"], " \\\\")
  })
  
  
  # Print LaTeX table
  cat("{\\renewcommand{\\arraystretch}{0.9}\n")
  cat("\\begin{table}[h!]\n\\centering\n\\scriptsize\n")
  cat("\\begin{tabular}{lrrr|rrr}\n")
  cat("& \\multicolumn{3}{c}{Uniform Mixture} & \\multicolumn{3}{c}{Gaussian Mixture}\\\\\n")
  cat("p/n & $d=0$ & $d=2$ & $d=4$ & $d=0$ & $d=2$ & $d=4$ \\\\ \\hline\n")
  cat(paste(rows, collapse="\n"), "\n")
  cat("\\end{tabular}\n")
  cat(sprintf("\\caption{Average Cram\'{e}r–Wold distance over 30 trials for %s across different $d$ values (standard deviation in parentheses).}\n", method_name))
  cat(sprintf("\\label{tab:%s_tv_errors}\n", tolower(method_name)))
  cat("\\end{table}}\n")
}



# Run for your table:
make_latex_table_compare(results_mcluster, "mcluster")
make_latex_table_compare(results_kdvine, "kdvine")
make_latex_table_compare(results_band, "band")
make_latex_table_compare(results_nflow, "nflow")
