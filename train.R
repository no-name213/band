# rm(list = ls())
# ==============================================================================
# BAND Framework: Training & Model Interfaces
# Description: Implements Binned Autoregressive Nonparametric Distributions
#              using CatBoost for multivariate conditional modeling.
# ==============================================================================


# Dependencies
library(catboost)
# CatBoost installation (if needed):
#   install.packages('remotes')
#   remotes::install_github('catboost/catboost',
#                           subdir = 'catboost/R-package',
#                           INSTALL_opts = '--no-staged-install')
library(here)

source(here("utils.R"))
source(here("sampling.R"))

#' CatBoost Training Generator
#'
#' Factory function that returns a training function with fixed hyperparameters.
#'
#' @param depth Tree depth (default 6).
#' @param l2_leaf_reg L2 regularization coefficient (default 3).
#' @param iterations Number of boosting iterations (default 1000).
#' @param bagging_temperature Strength of Bayesian bootstrap (default 1).
#' @return A function that takes (X, y) and returns a trained CatBoost model.
catboost_generator <- function(depth = 6, l2_leaf_reg = 3, iterations = 1000, bagging_temperature = 1) {
  
  train_catboost <- function(X, y) {
    if (!requireNamespace("catboost", quietly = TRUE)) stop("CatBoost not found.")
    
    # Stabilize regression targets with micro-noise
    y_stable <- y + rnorm(length(y), mean = 0, sd = 1e-16)
    train_pool <- catboost::catboost.load_pool(data = X, label = y_stable)
    
    params <- list(
      loss_function       = "RMSE",
      iterations          = as.integer(iterations),
      learning_rate       = 0.05,
      depth               = as.integer(depth),
      l2_leaf_reg         = l2_leaf_reg,
      bagging_temperature = bagging_temperature,
      random_strength     = 1,
      bootstrap_type      = "Bayesian",
      verbose             = 0,
      allow_writing_files = FALSE
    )
    
    return(catboost::catboost.train(learn_pool = train_pool, params = params))
  }
  return(train_catboost)
}

#' Prediction Wrapper
#'
#' @param X Predictor matrix.
#' @param model A trained CatBoost model object.
#' @return A numeric vector of raw predictions.
predict_catboost <- function(X, model) {
  test_pool <- catboost::catboost.load_pool(data = X)
  preds <- catboost::catboost.predict(model = model, pool = test_pool, prediction_type = "RawFormulaVal")
  return(as.numeric(preds))
}

#' Conditional Model Trainer
#'
#' Fits a sequence of conditional models for all variables in a dataset.
#'
#' @param x Input data (n x p matrix or data frame).
#' @param bin_all Binning metadata from bin_continuous.
#' @param model_info List containing 'train' and 'predict' functions.
#' @param verbose Logical; if TRUE, prints progress updates.
#' @return A nested list of models for each variable and its bins.
train_models <- function(x, bin_all, model_info, verbose = TRUE) {
  n <- nrow(x)
  p <- ncol(x)
  h_onehot <- x_onehot(x, bin_all)
  models <- vector("list", p)
  
  for (j in 1:p) {
    if (verbose) cat(sprintf("Training variable %d/%d\n", j, p))
    
    H_j <- h_onehot[[j]]
    m_j <- ncol(H_j)
    models[[j]] <- vector("list", m_j)
    
    if (j == 1) {
      for (i in 1:m_j) models[[j]][[i]] <- mean(H_j[, i])
    } else {
      X_predictors <- do.call(cbind, h_onehot[1:(j-1)])
      for (i in 1:m_j) {
        models[[j]][[i]] <- model_info$train(X_predictors, H_j[, i])
      }
    }
  }
  return(models)
}

#' BAND Main Interface
#'
#' Higher-level function to train a BAND model and return a sampler.
#'
#' @param x Input dataset.
#' @param factor Smoothing factor for bin calculation (default 2).
#' @param need_presampling Logical; if TRUE, pre-calculates unique hypercubes.
#' @return A sampler function with model metadata attached as attributes.
band <- function(x, factor = 2, need_presampling = TRUE) {
  if (!is.matrix(x)) x <- matrix(x, ncol = 1)
  n <- nrow(x)
  
  # Initialize models
  train_fn <- catboost_generator(depth = 6, l2_leaf_reg = 3)
  model_info <- list(train = train_fn, predict = predict_catboost)
  
  # Binning & Training
  k <- ceiling(factor * n^(1/3))
  bin_all <- bin_continuous(x, k = k)
  models_ <- train_models(x, bin_all, model_info, verbose = FALSE)
  
  # Optimization
  factor_result <- find_best_factor(models_, model_info, bin_all, x)
  factor_best   <- factor_result$factor_best
  
  # Define Closure
  band_sampler <- function(n_sampling, factor = factor_best, onehot = FALSE) {
    h_onehot_sampling <- sampling_onehot_model(n_sampling, models_, model_info, factor)
    if (onehot) return(h_onehot_sampling)
    return(onehot_x(h_onehot_sampling, bin_all))
  }
  
  attr(band_sampler, "model_info")  <- model_info
  attr(band_sampler, "bin_all")     <- bin_all
  attr(band_sampler, "models_")     <- models_
  attr(band_sampler, "factor_best") <- factor_best
  # Unique Key Discovery (Pre-sampling)
  if (need_presampling) {
    batch <- 100000
    global_keys <- character(0)
    L_unique_old <- 0
    increment <- 1
    h_onehot_temp <- vector("list", ncol(x))
    
    while (increment >= ceiling(L_unique_old * 0.0005)) {
      h_oh <- band_sampler(batch, factor = factor_best, onehot = TRUE)
      
      # h_oh is a LIST. do.call(cbind, h_oh) creates the matrix for key generation
      keys <- apply(do.call(cbind, h_oh), 1, paste, collapse = "-")
      
      unique_mask <- !duplicated(keys)
      new_mask <- !(keys[unique_mask] %in% global_keys)
      final_idx <- which(unique_mask)[new_mask]
      
      if (length(final_idx) > 0) {
        for (j in seq_len(ncol(x))) {
          # FIX: Access the j-th matrix in the list h_oh before indexing rows
          h_onehot_temp[[j]] <- rbind(h_onehot_temp[[j]], 
                                      h_oh[[j]][final_idx, , drop = FALSE])
        }
        
        global_keys <- c(global_keys, keys[final_idx])
      }
      
      increment <- length(global_keys) - L_unique_old
      L_unique_old <- length(global_keys)
      
      # Safety break: if no new keys are found at all, stop to avoid infinite loop
      if (increment == 0) break 
    }
    attr(band_sampler, "h_onehot") <- h_onehot_temp
  }
  
  return(band_sampler)
}

#' Optimize Sampling Factor
#'
#' Evaluates different thresholding factors to minimize Cramér–Wold distance.
#'
#' @param models_ Trained list of models.
#' @param model_info Training/Prediction metadata.
#' @param bin_all Binning specifications.
#' @param x Original data matrix for comparison.
#' @param factors Vector of factors to test.
#' @param n_sample Number of samples used for evaluation.
#' @return A list containing the 'factor_best' and 'err_best'.
find_best_factor <- function(models_, model_info, bin_all, x, 
                             factors = c(0, 0.05, 0.1, 0.3, 1.0), n_sample = 1000) {
  best_factor <- factors[1]
  best_err <- Inf
  
  for (fac in factors) {
    x_samp <- onehot_x(sampling_onehot_model(n_sample, models_, model_info, fac), bin_all)
    err <- cramer_wold_distance(x_samp, x, N = n_sample)
    if (err < best_err) {
      best_err <- err
      best_factor <- fac
    }
  }
  return(list(factor_best = best_factor, err_best = best_err))
}