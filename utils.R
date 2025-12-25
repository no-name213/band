# ==============================================================================
# BAND Framework: Utility Modules
# Description: Helper functions for binning, one-hot encoding/decoding, 
#              and multivariate distance metrics.
# ==============================================================================

library(here)
source(here("sampling.R"))

# ==============================================================================
# 1. Binning & Interval Parsing
# ==============================================================================

#' Parse Interval Strings
#'
#' Converts string representations like "(-10, -1]" into numeric bounds.
#'
#' @param s A string representing a mathematical interval.
#' @return A list containing numeric bounds 'a', 'b' and logicals 'left_open', 'right_open'.
parse_interval <- function(s) {
  s <- gsub("\\s+", "", s)
  left_open  <- substr(s, 1, 1)  == "("
  right_open <- substr(s, nchar(s), nchar(s)) == ")"
  
  inside <- substring(s, 2, nchar(s) - 1)
  parts  <- strsplit(inside, ",")[[1]]
  
  return(list(
    a = as.numeric(parts[1]), 
    b = as.numeric(parts[2]),
    left_open = left_open,
    right_open = right_open
  ))
}

#' Continuous Data Binning
#'
#' Creates a set of intervals (bins) for each column in a dataset.
#'
#' @param x Input vector or matrix.
#' @param k Number of bins (default 4).
#' @param regular Logical; if TRUE, the first bin is closed on the left.
#' @param margin Small numeric offset to ensure edge cases fall within bins.
#' @return A list of lists containing string-formatted intervals for each dimension.
bin_continuous <- function(x, k = 4, regular = TRUE, margin = 1e-4) {
  if (!is.matrix(x)) x <- matrix(x, ncol = 1)
  p <- ncol(x)
  bin_all <- vector("list", p)
  
  for (j in 1:p) {
    xj <- x[, j]
    breaks <- seq(from = min(xj), to = max(xj), length.out = k)
    
    if (max(breaks) < max(xj)) breaks <- c(breaks, max(xj))
    
    # Expand boundaries slightly for numerical stability
    breaks[1] <- breaks[1] - margin
    breaks[length(breaks)] <- breaks[length(breaks)] + margin
    
    bins_j <- vector("list", length(breaks) - 1)
    for (k_ in seq_len(length(breaks) - 1)) {
      left  <- breaks[k_]
      right <- breaks[k_ + 1]
      
      left_bracket  <- if (regular && k_ == 1) "[" else "("
      right_bracket <- "]"
      
      # Use fixed precision to prevent string shortcutting
      bins_j[[k_]] <- sprintf("%s%.10f, %.10f%s", left_bracket, left, right, right_bracket)
    }
    bin_all[[j]] <- bins_j
  }
  return(bin_all)
}

#' Center Subcube Calculation
#'
#' Shrinks an interval to a desired volume ratio while maintaining the center.
#'
#' @param a Lower bound.
#' @param b Upper bound.
#' @param ratio Desired volume ratio.
#' @param p Dimension of the full cube.
#' @return Numeric vector with 'a_new' and 'b_new'.
make_centered_subcube <- function(a, b, ratio, p) {
  s     <- ratio^(1/p)
  L     <- b - a
  L_new <- s * L
  mid   <- (a + b) / 2
  
  return(c(a_new = mid - L_new / 2, b_new = mid + L_new / 2))
}

# ==============================================================================
# 2. Encoding & Decoding (One-Hot)
# ==============================================================================

#' One-Hot Encoding
#'
#' Maps continuous values into binary one-hot matrices based on provided bins.
#'
#' @param x Input data matrix.
#' @param bin_all Binning metadata.
#' @param ratio Optional; volume ratio for subcube adjustment.
#' @param x_adj_coordinates Optional; vector indicating which coordinates to adjust.
#' @return A list of one-hot matrices, one per dimension.
x_onehot <- function(x, bin_all, ratio = NA, x_adj_coordinates = NA) {
  if (!is.matrix(x)) x <- matrix(x, nrow = 1)
  n <- nrow(x)
  p <- ncol(x)
  h_onehot <- vector("list", p)
  
  for (j in 1:p) {
    bins_j     <- lapply(bin_all[[j]], parse_interval)
    mj         <- length(bins_j)
    h          <- matrix(0, n, mj)
    xj         <- x[, j]
    
    a_vec      <- sapply(bins_j, function(b) b$a)
    b_vec      <- sapply(bins_j, function(b) b$b)
    left_open  <- sapply(bins_j, function(b) b$left_open)
    right_open <- sapply(bins_j, function(b) b$right_open)
    
    in_bin <- mapply(function(a, b, lo, ro) {
      if (!is.na(ratio) && x_adj_coordinates[j] == 1) {
        A <- make_centered_subcube(a, b, ratio, sum(x_adj_coordinates))
        a <- A[1]; b <- A[2]
      }
      left_ok  <- if (lo) xj > a else xj >= a
      right_ok <- if (ro) xj < b else xj <= b
      left_ok & right_ok
    }, a_vec, b_vec, left_open, right_open)
    
    if (!is.matrix(in_bin)) in_bin <- matrix(in_bin, nrow = 1)
    
    idx <- max.col(in_bin, ties.method = "first")
    idx[rowSums(in_bin) == 0] <- NA 
    
    h[cbind(1:n, idx)] <- 1
    h_onehot[[j]] <- h
  }
  return(h_onehot)
}

#' One-Hot Decoding
#'
#' Maps binary one-hot matrices back to continuous values via uniform sampling within bins.
#'
#' @param h_onehot List of one-hot matrices.
#' @param bin_all Binning metadata.
#' @return A matrix of continuous values.
onehot_x <- function(h_onehot, bin_all) {
  h_onehot <- lapply(h_onehot, function(h) if (is.vector(h)) matrix(h, nrow = 1) else h)
  p <- length(h_onehot)
  n <- nrow(h_onehot[[1]])
  x <- matrix(0, n, p)
  
  for (j in 1:p) {
    h      <- h_onehot[[j]]
    bins_j <- lapply(bin_all[[j]], parse_interval)
    idx    <- max.col(h, ties.method = "first")
    
    # Fallback: Sample randomly if no bin is active
    zero_rows <- rowSums(h) == 0
    if (any(zero_rows)) idx[zero_rows] <- sample(ncol(h), sum(zero_rows), replace = TRUE)
    
    a_vec <- sapply(bins_j, `[[`, "a")
    b_vec <- sapply(bins_j, `[[`, "b")
    
    x[, j] <- runif(n, min = a_vec[idx], max = b_vec[idx])
  }
  return(x)
}

# ==============================================================================
# 3. Metrics & Diagnostics
# ==============================================================================

#' Cramér–Wold Distance
#'
#' Computes the maximum discrepancy between two multivariate samples using 
#' sparse, random, and axial projections.
#'
#' @param x Sample matrix A.
#' @param y Sample matrix B.
#' @param N Number of random directions.
#' @param z_step Grid step size for CDF evaluation.
#' @return The maximum Kolmogorov-Smirnov distance across all projections.
cramer_wold_distance <- function(x, y, N = 10000, z_step = 0.02) {
  stopifnot(ncol(x) == ncol(y))
  p <- ncol(x); x <- as.matrix(x); y <- as.matrix(y)
  z_grid <- seq(from = -20, to = 20, by = z_step)
  
  # 1. Sparse Projections (Coordinate Pairs)
  proj_sparse <- matrix(0, nrow = N, ncol = p)
  for (i in 1:N) {
    indices <- sample(p, 2, replace = FALSE)  
    proj_sparse[i, indices] <- stats::rnorm(2)
  }
  proj_sparse <- proj_sparse / sqrt(rowSums(proj_sparse^2))
  
  # 2. Dense Random Projections
  proj_dense <- matrix(rnorm(N * p), nrow = N, ncol = p)
  proj_dense <- proj_dense / sqrt(rowSums(proj_dense^2))
  
  # Combine Projections: Sparse + Dense + Axial (Identity)
  x_proj <- cbind(x %*% t(proj_sparse), x %*% t(proj_dense), x)
  y_proj <- cbind(y %*% t(proj_sparse), y %*% t(proj_dense), y)
  
  K <- ncol(x_proj)
  cw <- numeric(K)
  for (k in 1:K) {
    ex <- stats::ecdf(x_proj[, k])
    ey <- stats::ecdf(y_proj[, k])
    cw[k] <- max(abs(ex(z_grid) - ey(z_grid)))
  }
  return(max(cw))
}

#' Joint Probability Prediction
#'
#' Calculates the joint probability for a given set of one-hot hypercubes.
#'
#' @param h_onehot List of one-hot encoded matrices.
#' @param band_sampler A trained BAND sampler function with model attributes.
#' @return A numeric vector of joint probabilities.
predict_prob_h_onehot <- function(h_onehot, band_sampler) {
  model_info <- attr(band_sampler, "model_info")
  models_    <- attr(band_sampler, "models_")
  p <- length(h_onehot)
  n <- nrow(h_onehot[[1]])
  
  predictions <- matrix(0, nrow = n, ncol = p)
  for (j in 1:p) {
    H_j <- if (!is.matrix(h_onehot[[j]])) matrix(h_onehot[[j]], nrow = 1) else h_onehot[[j]]
    m_j <- ncol(H_j)
    
    if (j == 1) {
      for (i in 1:m_j) {
        row_idx <- which(H_j[, i] == 1)
        if (length(row_idx) > 0) predictions[row_idx, j] <- models_[[j]][[i]]
      }
    } else {
      X_predictors <- do.call(cbind, h_onehot[1:(j-1)])
      for (q in 1:m_j) {
        row_idx <- which(H_j[, q] == 1)
        if (length(row_idx) > 0) {
          predictions[row_idx, j] <- model_info$predict(X_predictors[row_idx, , drop = FALSE], 
                                                        models_[[j]][[q]])
        }
      }
    }
  }
  return(apply(predictions, 1, prod))
}