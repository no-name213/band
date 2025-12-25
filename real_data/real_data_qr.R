# rm(list = ls())
# ==============================================================================
# Script: real_data_qr.R
# Description: Conducts real-data experiments using Linear Quantile Regression
#              (QR) as a baseline comparison for the BAND framework.
# Part 1: Empirical Coverage Evaluation (Bonferroni-adjusted)
# Part 2: Appendix Visualization Generation
# ==============================================================================

library(quantreg)
library(here)
source(here("confidence_region.R"))
source(here("real_data/data_process.R")) # Loads data_processed

# ------------------------------------------------------------------------------
# 1. Global Setup & Configuration
# ------------------------------------------------------------------------------

data <- as.matrix(data_processed)
SAVE <- TRUE

# Configuration matching real_data_band.R and real_data_qrf.R
COORDINATE_SET <- c("CPI_RATE", "FEDFUNDS_CHANGE", "UNRATE_CHANGE")

CUTOFF_SETS <- list(
  period1 = as.Date(c("2014-01-01", "2019-12-01")),
  period2 = as.Date(c("2022-08-01", "2025-08-01"))
)

coordinate_combinations <- list(
  c("CPI_RATE", "FEDFUNDS_CHANGE", "UNRATE_CHANGE"),
  c("CPI_RATE", "UNRATE_CHANGE"),
  c("CPI_RATE", "FEDFUNDS_CHANGE"),
  c("FEDFUNDS_CHANGE", "UNRATE_CHANGE"),
  c("CPI_RATE"),
  c("UNRATE_CHANGE"),
  c("FEDFUNDS_CHANGE")
)

ALPHA_LEVELS <- c(seq(0.1, 0.9, by = 0.1), 0.95, 0.99)
results      <- list()

# ------------------------------------------------------------------------------
# 2. Experiment Execution Loop
# ------------------------------------------------------------------------------

for (target_vars in coordinate_combinations) {
  variable_key <- paste(target_vars, collapse = "_")
  results[[variable_key]] <- list()
  
  for (lag in c(1, 3, 6)) {
    # Generate AR lag matrix
    lag_matrix <- create_ar_lag_matrix(data, COORDINATE_SET, c(0, lag))
    dates      <- as.Date(rownames(lag_matrix), format = "%m/%d/%Y")
    
    # Responses (Y) and Predictors (X)
    y_all <- lag_matrix[, 1:3] 
    x_all <- lag_matrix[, 4:6]
    
    df_full <- data.frame(
      date = dates,
      Y1 = y_all[,1], Y2 = y_all[,2], Y3 = y_all[,3],
      X1 = x_all[,1], X2 = x_all[,2], X3 = x_all[,3]
    )
    
    for (period_name in names(CUTOFF_SETS)) {
      cutoff_set <- CUTOFF_SETS[[period_name]]
      
      train_idx <- which(df_full$date < cutoff_set[1])
      test_idx  <- which(df_full$date >= cutoff_set[1] & df_full$date <= cutoff_set[2])
      
      df_train <- df_full[train_idx, ]
      df_test  <- df_full[test_idx, ]
      
      coverage_seq <- numeric(length(ALPHA_LEVELS))
      
      for (i in seq_along(ALPHA_LEVELS)) {
        alpha <- ALPHA_LEVELS[i]
        k     <- length(target_vars)
        
        # Bonferroni adjustment
        beta      <- 1 - (1 - alpha) / k
        tau_lower <- (1 - beta) / 2
        tau_upper <- 1 - tau_lower
        
        # Fit models for current alpha (fitted once per alpha level, not per row)
        lower_models <- list()
        upper_models <- list()
        
        for (col_name in target_vars) {
          col_ind <- which(COORDINATE_SET == col_name)
          formula_obj <- as.formula(paste0("Y", col_ind, " ~ X1 + X2 + X3"))
          
          lower_models[[col_name]] <- rq(formula_obj, data = df_train, tau = tau_lower)
          upper_models[[col_name]] <- rq(formula_obj, data = df_train, tau = tau_upper)
        }
        
        # Calculate coverage over test set
        hits <- 0
        for (row in 1:nrow(df_test)) {
          in_interval <- TRUE
          for (col_name in target_vars) {
            col_ind  <- which(COORDINATE_SET == col_name)
            y_actual <- df_test[row, paste0("Y", col_ind)]
            
            l_pred <- predict(lower_models[[col_name]], newdata = df_test[row, ])
            u_pred <- predict(upper_models[[col_name]], newdata = df_test[row, ])
            
            if (!(y_actual >= l_pred && y_actual <= u_pred)) {
              in_interval <- FALSE
              break
            }
          }
          if (in_interval) hits <- hits + 1
        }
        coverage_seq[i] <- hits / nrow(df_test)
      }
      
      results[[variable_key]][[period_name]][[paste0("lag_", lag)]] <- list(
        alpha    = ALPHA_LEVELS,
        coverage = coverage_seq
      )
      
      cat(sprintf("[%s] QR Var: %s | Lag: %d | Status: Complete\n", 
                  period_name, variable_key, lag))
    }
  }
}

# Save results
if (SAVE) {
  res_path <- here("real_data", "real_data_results", "qr.rds")
  dir.create(dirname(res_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(results, res_path)
  cat("QR results saved to:", res_path, "\n")
}



#######
## make plot:
file_path <- here::here('real_data', 'real_data_results', 'qr.rds')
results <- readRDS(file_path)


old_par <- par(no.readonly = TRUE)  # save current settings

lags <- c(1, 3, 6)
lag_col <- c("1" = "black", "3" = "blue", "6" = "red")
lag_lty <- c("1" = 1, "3" = 2, "6" = 3)

for (coordinate_set in names(results)) {
  for (period_name in names(results[[coordinate_set]])) {
    
    
    
    # File name for saving
    img_name <- paste0(
      paste(coordinate_set, collapse = "_"),
      "_period_", period_name, "_RQ_.png"
    )
    img_path <- here::here("real_data", 'qr', img_name)
    
    
    if (SAVE) png(img_path, width = 1300, height = 1300, res = 150)
    
    par(mar = c(5, 6, 4, 2))
    
    # Determine y-label
    # y_lab <- if (coordinate_set == "CPI_RATE" & period_name == "period1") "Empirical coverage" else ""
    y_lab <- ""
    
    # Start empty plot
    plot(NULL, NULL,
         xlim = c(0.1, 1.0),
         ylim = c(0, 1),
         xlab = "alpha",
         ylab = "",
         xaxt = "n",
         yaxt = "n",
         cex.lab = 2.4,
         cex.axis = 1.9)
    
    # Y-axis
    yticks <- seq(0, 1, 0.1)
    axis(2, at = yticks, las = 1, cex.axis = 1.7)
    
    # X-axis
    # X-axis: Sequence up to 0.95
    xticks_main <- c(seq(0.1, 0.9, 0.1), 0.95)
    axis(1, at = xticks_main, cex.axis = 1.7)
    
    # X-axis: Explicitly add the 0.99 tick
    # We draw a *second* axis call only for the 0.99 and 1.0 mark/label
    # lwd=0 suppresses the axis line, but ticks and labels are drawn.
    # A small negative tcl forces the tick to point inward, but you can omit it.
    axis(1, at = c(0.99), labels = c("0.99"), 
         cex.axis = 1.7, 
         lwd = 0,      # Suppress the redundant axis line
         lwd.ticks = 1 # Ensure the tick marks are drawn
    )
    
    # Add grid lines
    for (y in seq(0, 1, by = 0.1)) abline(h = y, col = rgb(0.5,0.5,0.5,0.3), lty = 3)
    for (x in seq(0, 1, by = 0.1)) abline(v = x, col = rgb(0.5,0.5,0.5,0.3), lty = 3)
    
    # 45-degree reference line
    abline(a = 0, b = 1, col = rgb(0.3,0.3,0.3,0.35), lwd = 2)
    
    # Plot each lag line
    for (lag in lags) {
      lag_name <- paste0("lag_", lag)
      if (!lag_name %in% names(results[[coordinate_set]][[period_name]])) next
      df <- results[[coordinate_set]][[period_name]][[lag_name]]
      lines(df$alpha, df$coverage, col = lag_col[as.character(lag)],
            lty = lag_lty[as.character(lag)], lwd = 2)
    }
    
    # Add legend
    legend("topleft",
           legend = paste0(lags, "-month forecast"),
           col = lag_col[as.character(lags)],
           lty = lag_lty[as.character(lags)],
           lwd = 2,
           bty = "n",
           cex = 1.5)
    
    # Add y-label manually
    mtext(y_lab, side = 2, line = 4.5, cex = 1.7)
    
    if (SAVE) dev.off()
    
  }
}



