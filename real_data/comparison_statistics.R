# rm(list = ls())
library(here)

# ==============================================================================
# Script: generate_latex_table.R
# ==============================================================================

### ----------------------------
# INITIALIZE EMPTY OUTCOME MATRIX
### ----------------------------
row_names <- c("Period_I_aveall", "Period_I_ave_09", "Period_II_aveall", "Period_II_ave_09")
col_names <- c(
  paste("BAND", paste0("col_", 1:2), sep = "_"),
  paste("QR",   paste0("col_", 1:2), sep = "_"),
  paste("QRF",  paste0("col_", 1:2), sep = "_")
)

results <- matrix(NA_real_, nrow = 4, ncol = length(col_names), dimnames = list(row_names, col_names))

### ----------------------------
# LOADING DATA FOR OUTCOME MATRIX
### ----------------------------
for (row_idx in c(1, 3)) {
  for (col_idx in c(1, 2)) {

    # Selection of coordinate combinations
    if (col_idx == 1) {
      coordinate_combinations <- c("CPI_RATE", "FEDFUNDS_CHANGE", "UNRATE_CHANGE")
    } else {
      coordinate_combinations <- list(
        c("CPI_RATE", "FEDFUNDS_CHANGE", "UNRATE_CHANGE"),
        c("CPI_RATE", "FEDFUNDS_CHANGE"),
        c("FEDFUNDS_CHANGE", "UNRATE_CHANGE"),
        c("CPI_RATE", "UNRATE_CHANGE")
      )
    }

    # Period and Tuning logic
    if (row_idx == 1) {
      FACTORS <- c("0.9")
      PERIOD <- c("period1")
    } else {
      FACTORS <- c("1")
      PERIOD <- c("period2")
    }

    # --- BAND Calculation ---
    file_path <- here::here("real_data", "real_data_results", "band.rds")
    loaded_results <- readRDS(file_path)
    ave_09 <- 0; accu <- 0; ind <- 0

    for (coordinate_set in coordinate_combinations) {
      variable_names <- paste(coordinate_set, collapse = "_")
      for (lag in c(1, 3, 6)) {
        for (factor in FACTORS) {
          for (period_name in PERIOD) {
            alpha_vec <- loaded_results[[variable_names]][[as.character(lag)]][[as.character(factor)]][[period_name]][, "alpha"]
            coverage_vec <- loaded_results[[variable_names]][[as.character(lag)]][[as.character(factor)]][[period_name]][, "empirical_coverage"]
            gap <- abs(alpha_vec - coverage_vec)
            ave_09 <- ave_09 + gap[9]
            accu <- accu + mean(gap[1:9])
            ind <- ind + 1
          }
        }
      }
    }
    results[row_idx, col_idx] <- accu / ind
    results[row_idx + 1, col_idx] <- ave_09 / ind

    # --- QR Calculation ---
    file_path <- here::here('real_data', 'real_data_results', 'qr.rds')
    loaded_results <- readRDS(file_path)
    ave_09 <- 0; accu <- 0; ind <- 0
    for (coordinate_set in coordinate_combinations) {
      for (lag in c(1, 3, 6)) {
        for (period_name in PERIOD) {
          target_name <- paste(coordinate_set, collapse = "_")
          coverage_vec <- loaded_results[[target_name]][[period_name]][[paste0("lag_", lag)]]$coverage
          alpha_vec <- loaded_results[[target_name]][[period_name]][[paste0("lag_", lag)]]$alpha
          gap <- abs(alpha_vec - coverage_vec)
          ave_09 <- ave_09 + gap[9]
          accu <- accu + mean(gap[1:9])
          ind <- ind + 1
        }
      }
    }
    results[row_idx, col_idx + 2] <- accu / ind
    results[row_idx + 1, col_idx + 2] <- ave_09 / ind

    # --- QRF Calculation ---
    file_path <- here::here('real_data', 'real_data_results', 'qrf.rds') # Fixed path name to lowercase
    loaded_results <- readRDS(file_path)
    ave_09 <- 0; accu <- 0; ind <- 0
    for (coordinate_set in coordinate_combinations) {
      for (lag in c(1, 3, 6)) {
        for (period_name in PERIOD) {
          target_name <- paste(coordinate_set, collapse = "_")
          coverage_vec <- loaded_results[[target_name]][[period_name]][[paste0("lag_", lag)]]$coverage
          alpha_vec <- loaded_results[[target_name]][[period_name]][[paste0("lag_", lag)]]$alpha
          gap <- abs(alpha_vec - coverage_vec)
          ave_09 <- ave_09 + gap[9]
          accu <- accu + mean(gap[1:9])
          ind <- ind + 1
        }
      }
    }
    results[row_idx, col_idx + 4] <- accu / ind
    results[row_idx + 1, col_idx + 4] <- ave_09 / ind
  }
}

### ----------------------------
# OUTPUT LATEX TABLE CONTENT
### ----------------------------
fmt <- function(x, digits = 3) {
  formatC(x, format = "f", digits = digits)
}

rows <- list(
  I_aveall  = results["Period_I_aveall", ],
  I_ave09   = results["Period_I_ave_09", ],
  II_aveall = results["Period_II_aveall", ],
  II_ave09  = results["Period_II_ave_09", ]
)

# Restore original 3-argument function
latex_row <- function(prefix, label, values) {
  paste0(prefix, " & ", label, " & ", paste(fmt(values), collapse = " & "), " \\\\")
}

# Restore original cat block in a single call as requested
cat(
  "% --- Auto-generated table body ---\n",
  "\\multirow{2}{*}{I}",
  latex_row("", "$\\alpha \\le 0.9$", rows$I_aveall), "\n",
  latex_row("", "$\\alpha = 0.9$", rows$I_ave09), "\n",
  "\\hline\n",
  "\\multirow{2}{*}{II}",
  latex_row("", "$\\alpha \\le 0.9$", rows$II_aveall), "\n",
  latex_row("", "$\\alpha = 0.9$", rows$II_ave09), "\n",
  sep = ""
)
