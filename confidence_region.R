# rm(list = ls())
# ==============================================================================
# BAND Framework: Confidence Regions
# Description: Implements the construction of high-density confidence regions 
#              by identifying the smallest set of hypercubes covering a 
#              specified probability mass.
# ==============================================================================

library(here)


source(here("utils.R"))
source(here("train.R"))

#' Compute BAND Confidence Regions
#'
#' Finds the smallest set of hypercubes that contain at least alpha_level 
#' of the probability mass, potentially conditioned on observed variables.
#'
#' @param band_sampler A trained BAND sampler function with model attributes.
#' @param x Vector or matrix of values to condition on (use NA for coordidantes not to conditional on).
#' @param x_ignore Vector of indices (0/1) to marginalize out of the region.
#' @param alpha_level The desired confidence level (default 0.9).
#' @return A list containing the selected one-hot cubes, group indices, 
#'         and interpolation parameters for the boundary cube.
confidence_region <- function(band_sampler, 
                              x_condi = NA, 
                              x_ignore = NA, 
                              alpha_level = 0.9) {
  
  required_attrs <- c("bin_all", "models_", "model_info", "h_onehot")
  for (a in required_attrs) {
    if (is.null(attr(band_sampler, a))) {
      stop(paste("Missing required attribute:", a))
    }
  }
  
  # 1. Initialization & Metadata Extraction
  bin_all  <- attr(band_sampler, "bin_all")
  p        <- length(bin_all)
  h_stored <- attr(band_sampler, "h_onehot")
  
  if (all(is.na(x_condi))) x_condi <- rep(NA, p)
  if (!is.matrix(x_condi)) x_condi <- matrix(x_condi, nrow = 1)
  if (all(is.na(x_ignore))) x_ignore <- rep(0, p)
  
  # Convert conditional target into one-hot (NA coordinates become zero vectors)
  h_target <- x_onehot(x_condi, bin_all)
  
  # 2. Filtering: Find Pre-sampled Cubes Matching Conditional Constraints
  matching_idx <- seq_len(nrow(h_stored[[1]]))
  
  for (j in seq_len(p)) {
    # Match if coordinate is NA (sum == 0) OR if it matches target exactly
    idx_j <- which(apply(h_stored[[j]], 1, function(row) {
      all(row == h_target[[j]]) || sum(h_target[[j]]) == 0
    }))
    matching_idx <- intersect(matching_idx, idx_j)
  }
  
  # Fallback to unconditional if conditional set is empty
  if (length(matching_idx) == 0) {
    h_filtered <- h_stored
  } else {
    h_filtered <- lapply(h_stored, function(m) m[matching_idx, , drop = FALSE])
  }
  
  # 3. Probability Prediction
  # Calculate joint probabilities for the filtered cubes
  probs <- predict_prob_h_onehot(h_filtered, band_sampler)
  n_filtered <- nrow(h_filtered[[1]])
  
  # 4. Marginalization (Handling x_ignore)
  # Collapse probabilities over coordinates marked for ignoring
  h_mod <- h_filtered
  for (j in seq_len(p)) {
    if (x_ignore[j] == 1) h_mod[[j]] <- matrix(0, 
                                               n_filtered, 
                                               ncol(h_filtered[[j]]))
  }
  
  # Create signatures to detect unique patterns after ignoring dimensions
  h_combined <- do.call(cbind, h_mod)
  row_keys   <- apply(h_combined, 1, paste0, collapse = "_")
  uniq_keys  <- unique(row_keys)
  K          <- length(uniq_keys)
  
  # Aggregate probabilities for unique marginalized patterns
  h_new <- lapply(seq_len(p), function(j) matrix(0, K, ncol(h_filtered[[j]])))
  probs_new <- numeric(K)
  
  for (k in seq_len(K)) {
    idx <- which(row_keys == uniq_keys[k])
    row_rep <- h_combined[idx[1], ] # Representative pattern
    
    col_start <- 1
    for (j in seq_len(p)) {
      mj <- ncol(h_filtered[[j]])
      h_new[[j]][k, ] <- row_rep[col_start:(col_start + mj - 1)]
      col_start <- col_start + mj
    }
    probs_new[k] <- sum(probs[idx])
  }
  
  # 5. Selection: High Density Region Construction
  ordered_idx  <- order(probs_new, decreasing = TRUE)
  sorted_probs <- probs_new[ordered_idx]
  cum_probs    <- cumsum(sorted_probs) / sum(probs_new) # Normalized CDF
  
  # Identify boundary index
  used_k <- which(cum_probs >= alpha_level)[1]
  
  # Boundary Interpolation Logic
  c1 <- if (used_k > 1) cum_probs[used_k - 1] else 0
  c2 <- cum_probs[used_k]
  ratio <- (alpha_level - c1) / (c2 - c1)
  
  # Final Cube Extraction
  used_idx <- ordered_idx[1:used_k]
  h_output <- lapply(h_new, function(m) m[used_idx, , drop = FALSE])
  
  return(list(
    h_onehot   = h_new,           # All available marginalized cubes
    group_list = list(used_idx),  # Indices of cubes within alpha_level
    parts      = c(ordered_idx[used_k], ratio) # Boundary cube index and weight
  ))
}





#' Assign Cluster Labels to Samples
#'
#' Determines if samples fall within the constructed confidence regions, 
#' accounting for volume interpolation at the boundary.
#'
#' @param x Sample matrix to be labeled.
#' @param Z Result object from confidence_region().
#' @param band_sampler The trained BAND sampler containing binning metadata.
#' @param x_ignore Vector (0/1) of marginalized coordinates.
#' @param x_condi_coordinates Vector (0/1) of conditioned coordinates.
#' @return A numeric vector of cluster labels (0 for points outside the region).
cluster_labels <- function(x, 
                           Z, 
                           band_sampler, 
                           x_ignore = NA, 
                           x_condi_coordinates = NA) {
  
  # 1. Extract Metadata and Parameters
  h_onehot   <- Z$h_onehot
  group_list <- Z$group_list
  idx_boundary  <- Z$parts[1]
  vol_ratio     <- Z$parts[2]
  
  p <- length(h_onehot)
  bin_all <- attr(band_sampler, "bin_all")
  
  if (all(is.na(x_ignore))) x_ignore <- rep(0, p)
  if (all(is.na(x_condi_coordinates))) x_condi_coordinates <- rep(0, p)
  
  # 2. Encode Samples
  # Standard encoding
  X_encoded <- do.call(cbind, x_onehot(x, bin_all))
  
  # Volume-adjusted encoding (shrunk sub-cube) for the boundary case
  # Dimensions to adjust are those that are neither ignored nor conditioned on
  x_adj_coords <- 1 - x_ignore - x_condi_coordinates
  X_encoded_adj <- do.call(cbind, x_onehot(x, bin_all, vol_ratio, x_adj_coords))
  
  # Full hypercube matrix for reference
  H_full <- do.call(cbind, h_onehot)
  
  # Prepare column mask to ignore marginalized dimensions
  group_sizes <- sapply(h_onehot, ncol)
  col_mask    <- rep(x_ignore, group_sizes)
  
  # 3. Membership Assignment
  labels <- rep(0, nrow(X_encoded))
  
  # In our current application, there is actually only a single group.
  for (g_idx in seq_along(group_list)) {
    indices <- group_list[[g_idx]]
    
    # Check if samples fall into any cube within the current group
    # Logic: Match only on coordinates where col_mask == 0
    in_group <- apply(X_encoded, 1, function(row_x) {
      any(apply(H_full[indices, , drop = FALSE], 1, function(row_h) {
        all(row_h[col_mask == 0] == row_x[col_mask == 0])
      }))
    })
    
    labels[in_group] <- g_idx
    
    # 4. Boundary Adjustment (Interpolation)
    # Samples falling into the 'last' cube must be checked against the shrunk sub-cube
    is_at_boundary <- apply(X_encoded, 1, function(row_x) {
      all(H_full[idx_boundary, col_mask == 0] == row_x[col_mask == 0])
    })
    
    is_at_boundary_adj <- apply(X_encoded_adj, 1, function(row_x) {
      all(H_full[idx_boundary, col_mask == 0] == row_x[col_mask == 0])
    })
    
    # If a sample is in the boundary cube but NOT in the adjusted sub-cube, remove its label
    invalid_boundary <- is_at_boundary & !is_at_boundary_adj
    labels[invalid_boundary] <- 0
  }
  
  return(labels)
}

