# ==============================================================================
# BAND Framework: Sampling Modules
# Description: Generates synthetic data by sampling from learned conditional 
#              distributions.
# ==============================================================================

#' Sample One-Hot Vectors from Trained Models
#'
#' This function generates N samples of one-hot encoded vectors based on a 
#' sequence of conditional probability models.
#'
#' @param N Number of samples to generate.
#' @param models List of trained conditional models (from train_models).
#' @param model_info List containing the custom 'predict' function.
#' @param factor Thresholding factor for probability pruning (default 0.0).
#' @return A list of length p, where each element is an N x m_j one-hot matrix.
sampling_onehot_model <- function(N, models, model_info, factor = 0.0) {
  
  p <- length(models)
  h_onehot <- vector("list", p)
  
  for (j in 1:p) {
    m_j <- length(models[[j]])
    h_onehot[[j]] <- array(0, c(N, m_j))
    
    # --- Case 1: Unconditional Sampling (First Variable) ---
    if (j == 1) {
      probs <- unlist(models[[j]])
      probs <- probs / sum(probs) # Normalization
      
      # Sample indices based on empirical distribution
      idx <- sample(seq_along(probs), size = N, replace = TRUE, prob = probs)
      h_onehot[[j]][cbind(1:N, idx)] <- 1
      
    } else {
      # --- Case 2: Conditional Sampling (Subsequent Variables) ---
      
      # Prepare predictor matrix from all previously sampled variables
      X_predictors <- do.call(cbind, h_onehot[1:(j-1)])
      
      # Compute probability transition matrix across all bins
      prob_matrix <- array(0, c(N, m_j))
      for (q in 1:m_j) {
        prob_matrix[, q] <- model_info$predict(X_predictors, models[[j]][[q]])
      }
      
      # 1. Clean and Threshold Probabilities
      prob_matrix[prob_matrix < 0] <- 0  # Clip negative predictions
      
      # Apply dynamic thresholding based on row-wise standard deviation
      row_sd <- apply(prob_matrix, 1, sd)
      prob_matrix[prob_matrix < factor * row_sd[row(prob_matrix)]] <- 0
      
      # 2. Handle Zero-Probability Rows (Fallback to Uniform)
      zero_rows <- which(rowSums(prob_matrix) == 0)
      if (length(zero_rows) > 0) {
        prob_matrix[zero_rows, ] <- 1 / m_j
      }
      
      # 3. Normalize Rows
      prob_matrix <- prob_matrix / rowSums(prob_matrix)
      
      # 4. Vectorized Multinomial Sampling
      # Generate cumulative probabilities and compare against uniform random noise
      cumP <- t(apply(prob_matrix, 1, cumsum))
      u <- runif(N)
      
      # Identify the first column where cumulative probability exceeds the random draw
      idx <- max.col(cumP >= u, ties.method = "first")
      h_onehot[[j]][cbind(1:N, idx)] <- 1
    }
  }
  
  return(h_onehot)
}