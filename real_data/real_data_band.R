# rm(list = ls())
# ==============================================================================
# Script: real_data_band.R
# Description: Conducts real-data experiments using the BAND framework on 
#              FRED-MD macroeconomic data. Includes tuning and production phases.
# Part 1: Empirical Coverage Evaluation
# Part 2: Appendix Visualization Generation
# ==============================================================================

library(here)
source(here("confidence_region.R"))
source(here("real_data/data_process.R")) # Loads data_processed

# ------------------------------------------------------------------------------
# 1. Global Setup & Configuration
# ------------------------------------------------------------------------------

data <- as.matrix(data_processed)
cat("Data Loaded. Dimensions:", dim(data)[1], "rows,", dim(data)[2], "cols\n")

# Logic Controls
SAVE   <- TRUE
TUNING <- FALSE  # Set to TRUE to run tuning iterations

if (TUNING) {
  SAVE <- FALSE
  CUTOFF_SETS <- list(
    period3 = as.Date(c("2010-01-01", "2013-12-01")),
    period4 = as.Date(c("2021-06-01", "2022-07-01"))
  )
  ignored_sets <- list(c(""))
  LAGS <- c(6)
  FACTORS <- seq(0.4, 2.0, by = 0.1)
} else {
  CUTOFF_SETS <- list(
    period1 = as.Date(c("2014-01-01", "2019-12-01")),
    period2 = as.Date(c("2022-08-01", "2025-08-01"))
  )
  ignored_sets <- list(
    c(""),
    c("CPI_RATE"),
    c("FEDFUNDS_CHANGE"),
    c("UNRATE_CHANGE"),
    c("CPI_RATE", "FEDFUNDS_CHANGE"),
    c("FEDFUNDS_CHANGE", "UNRATE_CHANGE"),
    c("CPI_RATE", "UNRATE_CHANGE")
  )
  LAGS    <- c(1, 3, 6)
  FACTORS <- c(0.9, 1.0) 
}

ALPHA_LEVELS   <- c(seq(0.1, 0.9, by = 0.1), 0.95, 0.99)
COORDINATE_SET <- c("CPI_RATE", "FEDFUNDS_CHANGE", "UNRATE_CHANGE")
N              <- length(COORDINATE_SET)
trained_models <- list() # Cache for trained samplers
results        <- list()

# ------------------------------------------------------------------------------
# 2. Experiment Execution Loop
# ------------------------------------------------------------------------------

for (ignored_vars in ignored_sets) {
  
  # Identify active variables for this iteration
  not_used_idx   <- as.numeric(COORDINATE_SET %in% ignored_vars)
  active_vars    <- COORDINATE_SET[not_used_idx == 0]
  variable_key   <- paste(active_vars, collapse = "_")
  
  results[[variable_key]] <- list()
  
  for (lag in LAGS) {
    results[[variable_key]][[as.character(lag)]] <- list()
    
    for (factor in FACTORS) {
      results[[variable_key]][[as.character(lag)]][[as.character(factor)]] <- list()
      
      for (period_name in names(CUTOFF_SETS)) {
        
        # Initialize results matrix: [Alpha | Coverage]
        results[[variable_key]][[as.character(lag)]][[as.character(factor)]][[period_name]] <- matrix(
          NA_real_, nrow = length(ALPHA_LEVELS), ncol = 2,
          dimnames = list(NULL, c("alpha", "empirical_coverage"))
        )
        results[[variable_key]][[as.character(lag)]][[as.character(factor)]][[period_name]][, "alpha"] <- ALPHA_LEVELS
        
        # --- Data Splitting ---
        cutoff_set   <- CUTOFF_SETS[[period_name]]
        lag_matrix   <- create_ar_lag_matrix(data, COORDINATE_SET, c(0, lag))
        dates_lag    <- as.Date(rownames(lag_matrix), format = "%m/%d/%Y")
        
        start_idx    <- which(dates_lag >= cutoff_set[1])[1]
        end_idx      <- which(dates_lag >= cutoff_set[2])[1]
        
        x_train      <- lag_matrix[1:(start_idx - 1), ]
        x_test       <- lag_matrix[start_idx:end_idx, ]
        
        # --- Model Retrieval / Training ---
        model_key <- paste0("lag", lag, "_factor", factor, "_period", period_name)
        if (is.null(trained_models[[model_key]])) {
          cat(sprintf("Training BAND: %s...\n", model_key))
          trained_models[[model_key]] <- band(x_train, factor = factor)
        }
        band_sampler <- trained_models[[model_key]]
        
        # --- Evaluation ---
        for (a_idx in seq_along(ALPHA_LEVELS)) {
          alpha_level <- ALPHA_LEVELS[a_idx]
          hits <- 0
          
          for (t in 1:nrow(x_test)) {
            # Condition on past (lagged) data
            x_condi  <- c(rep(NA, N), x_test[t, (N + 1):(2 * N)])
            x_ignore <- c(not_used_idx, rep(0, N))
            
            # Construct Confidence Region
            Z <- confidence_region(band_sampler, 
                                   x_condi = x_condi, 
                                   x_ignore = x_ignore, 
                                   alpha_level = alpha_level)
            
            # Check Membership
            label <- cluster_labels(x_test[t, ], 
                                    Z, 
                                    band_sampler, 
                                    x_ignore = x_ignore, 
                                    x_condi_coordinates = 1 - is.na(x_condi))
            
            if (label == 1) hits <- hits + 1
          }
          
          cov_val <- hits / nrow(x_test)
          results[[variable_key]][[as.character(lag)]][[as.character(factor)]][[period_name]][a_idx, "empirical_coverage"] <- cov_val
          
          cat(sprintf("[%s] Var: %s | Lag: %d | Alpha: %.2f | Coverage: %.4f\n", 
                      period_name, variable_key, lag, alpha_level, cov_val))
        }
      }
    }
  }
}

# Save results
if (SAVE) {
  res_path <- here("real_data", "real_data_results", "band.rds")
  dir.create(dirname(res_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(results, res_path)
  cat("Results saved to:", res_path, "\n")
}



### ------------------------------
# SAVE FIGURES
### ------------------------------
coordinate_combinations <- list(
  c("CPI_RATE", "FEDFUNDS_CHANGE", "UNRATE_CHANGE"),
  c("CPI_RATE", "UNRATE_CHANGE"),
  c("CPI_RATE", "FEDFUNDS_CHANGE"),
  c("FEDFUNDS_CHANGE", "UNRATE_CHANGE"),
  c("CPI_RATE"),
  c("UNRATE_CHANGE"),
  c("FEDFUNDS_CHANGE")
)


FACTORS <- c(0.9, 1)  # or 1 or 0.9


file_path <- here::here("real_data", "real_data_results", "band.rds")
results <- readRDS(file_path)

for (coordinate_set in coordinate_combinations) {
  
  
    
  variable_names <- paste(coordinate_set, collapse = "_")
  
  results[[variable_names]]
  
  for (factor in FACTORS) {
    for (period_num in 1:2) {
      
      # Output PNG
      img_name <- paste0(
        paste0("factor_", as.character(factor), "/", paste(coordinate_set, collapse = "_")),
        "_factor_", factor,
        "_period_", period_num, ".png"
      )
      
      img_path <- here::here("real_data", img_name)
      
      if (SAVE) {
        png(img_path, width = 1300, height = 1300, res = 150)
      }
      
      
      par(mar = c(5, 6, 4, 2))
      if (identical(coordinate_set, 
                    c("CPI_RATE", "FEDFUNDS_CHANGE", "UNRATE_CHANGE"))) {
        y_lab <- "empirical coverage"
        # par(mgp = c(3.5, 1, 0))
        
      } else {
        y_lab <- ""
        # par(old_par)
      }
      
      
      # Styles
      lag_lty <- c("1" = 1, "3" = 2, "6" = 3)
      lag_col <- c("1" = "black", "3" = "blue", "6" = "red")
      
      # ---- 3 PANELS: one fine level repeated for each of the 3 positions ----
      for (panel_index in 1:3) {
        
        plot(NULL, NULL,
             xlim = c(0.1, 1.0),
             ylim = c(0, 1),
             xlab = "alpha",
             ylab = "",
             yaxt = "n",
             xaxt = "n",
             cex.lab = 2.4,
             cex.axis = 1.9,
        )
        
        # Add y-label manually with spacing control
        mtext(y_lab, 
              side = 2,       # left side
              line = 4.5,     # distance from axis
              cex = 1.9)
        
        # Y-axis
        yticks <- seq(0, 1, 0.1)
        axis(2, at = yticks, las = 1, cex.axis = 1.7)
        
        # X-axis
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
        
        # Horizontal dotted grid
        for (y in seq(0, 1, 0.1)) {
          abline(h = y, col = rgb(0.5, 0.5, 0.5, 0.3), lty = 3, lwd = 1)
        }
        
        # Vertical dotted grid
        for (x in seq(0, 1, 0.1)) {
          abline(v = x, col = rgb(0.5, 0.5, 0.5, 0.3), lty = 3, lwd = 1)
        }
        
        # 45° line
        abline(a = 0, b = 1,
               col = rgb(0.3, 0.3, 0.3, 0.35),
               lwd = 2,
               lty = 1)
        
        # Settings
        lags <- c("1", "3", "6")
        period_to_plot <- ifelse(period_num == 2, "period2", "period1")
        
        # Plot each lag
        for (lag in c(1, 3, 6)) {
          # if (!lag %in% names(organized_results)) next
          # if (!as.character(factor) %in% names(organized_results[[lag]])) next
          # if (!period_to_plot %in% names(organized_results[[lag]][[as.character(factor)]])) next
          
          df <- results[[variable_names]][[as.character(lag)]][[as.character(factor)]][[period_to_plot]]
          
          lines(df[, 'alpha'], df[, 'empirical_coverage'],
                col = lag_col[[as.character(lag)]],
                lty = lag_lty[[as.character(lag)]],
                lwd = 2)
        }
        
        # Legend
        legend("topleft",
               legend = paste(lags, "-month forcast"),
               col = lag_col[lags],
               lty = lag_lty[lags],
               lwd = 2,
               bty = "n",
               seg.len = 4,
               cex = 1.9)
  
      } # end panel loop
      
      dev.off()
      
    } # end period loop
  } # end factor loop
} # end coordinate loop


