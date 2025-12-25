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


# 
# # for equal interval bins
# confidence_region <- function(band_sampler,
#                               x = NA,
#                               x_ignore = NA,
#                               alpha_level = 0.9) {
#   model_info <- attr(band_sampler, "model_info")
#   bin_all    <- attr(band_sampler, "bin_all")
#   models_    <- attr(band_sampler, "models_")
#   p <- length(bin_all)
#   
#   if (all(is.na(x))) {
#     x <- rep(NA, p)
#   }
#   
#   
#   # Some coordinates of x are NA.
#   # If all coordinates of x are NAs, the reported results are unconditional.
#   # Non NA in x are coordinates to be conditional on
#   if (!is.matrix(x)) { x <- matrix(x, nrow = 1) }
#   
#   # When a biiger model band_sampler is provided, some coordiantes may not be of
#   # interest. These coordinates are in the set `which(x_ignore)`
#   if (all(is.na(x_ignore))) {x_ignore  <- rep(0, p)}
#   
#   # Turn x into onehot vectors
#   # NA coordinates correspond to all zeros in the onehot
#   h_to_be_conditional <- x_onehot(x, bin_all)
#   
#   
#   
#   # Determine the number of bins per variable
#   m <- rep(0, p)
#   for (j in 1:p) {
#     m[j] <- length(bin_all[[j]])  # number of categories for variable j
#   }
#   
#   
#   h_onehot <- attr(band_sampler, "h_onehot")
#   h_onehot_temp <- h_onehot
#   
#   
#   # Keep onehots matching the conditional values
#   matching_idx <- seq_len(nrow(h_onehot[[1]]))  # start with all indices
#   for (j in seq_len(p)) {
#     # Find rows in x_random[[j]] identical to the target one-hot vector one_hot_matrices[[j]]
#     # Some one_hot_matrices[[j]]'s are NA
#     # all(array==NA) = TRUE
#     idx_j <- which(apply(h_onehot[[j]], 
#                          1, 
#                          function(row) all(row == h_to_be_conditional[[j]]) 
#                          || sum(h_to_be_conditional[[j]]) == 0))
#     
#     # Keep only indices that match across all dimensions
#     matching_idx <- intersect(matching_idx, idx_j)
#   }
#   
#   for (j in seq_len(p)) {
#     h_onehot[[j]] <- h_onehot[[j]][matching_idx, , drop = FALSE]
#   }
#   
#   
#   if (length(matching_idx) == 0) {
#     # print('Unconditional results reported as the conditional set is empty.')
#     h_onehot <- h_onehot_temp
#   }
#   L_unique <- nrow(h_onehot[[1]])
#   
#   
#   
#   # Calculate the probability for each cube
#   probs <- predict_prob_h_onehot(h_onehot, band_sampler)
#   
#   ###
#   ###
#   ###
#   # Collapse the probabilities of cubes over the coordinates indicated in `x_ignore`
#   # and reorganize them into new cubes with aggregated probabilities.
#   # For each new cube, all coordinate groups marked by `x_ignore` are set to zero
#   # in the corresponding entries of `h_onehot`.
#   
#   # h_onehot: list of length p, each matrix is n × m_j
#   # probs: length-n vector
#   # x_ignore: length-p vector of 0/1, 1 = ignore group
#   n <- nrow(h_onehot[[1]])
#   
#   # 1. Make a modified version respecting x_ignore:
#   h_onehot_mod <- h_onehot
#   for (j in seq_len(p)) {
#     if (x_ignore[j] == 1) {
#       h_onehot_mod[[j]] <- matrix(0, n, ncol(h_onehot[[j]]))
#     }
#   }
#   
#   # 2. Combine groups into one big matrix
#   h_temp <- do.call(cbind, h_onehot_mod)
#   
#   # 3. Create row signatures to detect unique patterns
#   row_keys <- apply(h_temp, 1, paste0, collapse = "_")
#   uniq_keys <- unique(row_keys)
#   K <- length(uniq_keys)
#   
#   # 4. Initialize new list of p matrices
#   h_onehot_new <- lapply(seq_len(p), function(j) {
#     matrix(0, K, ncol(h_onehot[[j]]))
#   })
#   
#   # 5. New probability vector
#   probs_new <- numeric(K)
#   
#   # 6. For each unique pattern, extract example row and build new matrices
#   for (k in seq_len(K)) {
#     idx <- which(row_keys == uniq_keys[k])
#     
#     # Use the first row as the representative pattern
#     row_rep <- h_temp[idx[1], ]
#     
#     # Split back into group matrices
#     col_start <- 1
#     for (j in seq_len(p)) {
#       mj <- ncol(h_onehot[[j]])
#       h_onehot_new[[j]][k, ] <- row_rep[col_start:(col_start + mj - 1)]
#       col_start <- col_start + mj
#     }
#     
#     # Sum probabilities
#     probs_new[k] <- sum(probs[idx])
#   }
#   
#   
#   
#   h_onehot <- h_onehot_new
#   probs <- probs_new
#   
#   ####
#   ####
#   ####
#   
#   # Select cubes whose cumulative probability crosses the 
#   # alpha_level threshold.
#   ordered_idx <- order(probs, decreasing = TRUE)
#   
#   sorted_probs <- probs[ordered_idx]
#   cum_probs <- cumsum(sorted_probs)
#   # Normalize cumulative probabilities to [0,1]
#   cum_probs <- cum_probs  / sum(probs)
#   
#   ###
#   ###
#   # Find first index where CDF >= alpha
#   used_k <- which(cum_probs >= alpha_level)[1]
# 
#   
#   ratio <- 1
#   c1 <- 0
#   if (used_k > 1) {
#     c1 <- cum_probs[used_k - 1]
#   } 
#   c2 <- cum_probs[used_k]
#   
#   # interpolate between cum_probs[k-1] and cum_probs[k]
#   ratio <- (alpha_level - c1) / (c2 - c1)
#   
#   
#   
#   # indices of selected cubes
#   used_idx <- ordered_idx[1:used_k]              
#   # Extract the corresponding one-hot vectors
#   h_onehot_output <- vector("list", p)
#   for (j in 1:p) {
#     h_onehot_output[[j]] <- h_onehot[[j]][used_idx, ]
#   }
#   # Output group list with a single group
#   group_list = list()
#   group_list[[1]] <- used_idx
#   return(list(h_onehot = h_onehot, 
#               group_list = group_list, 
#               parts = c(ordered_idx[used_k], ratio)))
# }



# 
# 
# 
# 
# # Check if the given sample is in the regions.
# cluster_labels <- function(x,
#                            Z,
#                            band_sampler,
#                            x_ignore = NA,
#                            x_condi_coordinates = NA) {
# 
#   h_onehot <- Z$h_onehot
#   group_list <- Z$group_list
#   ind <- Z$parts[1]
#   ratio <- Z$parts[2]
# 
# 
# 
#   p <- length(h_onehot)
#   bin_all <- attr(band_sampler, "bin_all")
#   if (all(is.na(x_ignore))) {
#     x_ignore <- rep(0, p)
#   }
# 
#   # Convert x into onehot vectors
#   # If a row is all zeros, the sample does not belong to any cluster
#   X_big <- do.call(cbind, x_onehot(x, bin_all))
# 
#   # Make the volume adjustment to the last cube,
#   # considering submodel applications.
#   x_adj_coordiantes <- 1 - x_ignore - x_condi_coordinates
#   X_big_adj <- do.call(cbind, x_onehot(x, bin_all, ratio, x_adj_coordiantes))
# 
#   H_big <- do.call(cbind, h_onehot)
# 
# 
#   cluster_labels <- rep(0, nrow(X_big))
#   for (idx in 1:length(group_list)) {
#     indices <- group_list[[idx]]
# 
#     group_sizes <- sapply(h_onehot, ncol)
#     col_ignore  <- rep(x_ignore, group_sizes)   # expanded mask
# 
#     index_in <- apply(X_big, 1, function(x) {
#       any(apply(H_big[indices, , drop = FALSE], 1, function(h) {
#         all(h[col_ignore == 0] == x[col_ignore == 0])
#       }))
#     })
# 
# 
# 
#     #
#     # # Assign group index to matching rows
#     cluster_labels[index_in] <- idx
# 
# 
# 
#     is_ind <- apply(X_big, 1, function(x) {
#       any(apply(H_big[ind, , drop = FALSE], 1, function(h) {
#         all(h[col_ignore == 0] == x[col_ignore == 0])
#       }))
#     })
# 
#     is_ind_adj <- apply(X_big_adj, 1, function(x) {
#       any(apply(H_big[ind, , drop = FALSE], 1, function(h) {
#         all(h[col_ignore == 0] == x[col_ignore == 0])
#       }))
#     })
#     original_labels <- cluster_labels
#     original_labels[is_ind] <- 0
#     # WORKING check this
#     original_labels[is_ind_adj] <- cluster_labels[is_ind_adj]
# 
#   }
#   return(original_labels)
# }
# 

# 
# 
# 
# 
# 
# ####
# ####
# ####
# ####
# 
# 
# # Form H_big via H_big <- do.call(cbind, h_onehot) and 
# # compute the number of communities 
# # (sets of cubes connected by any path).
# community_feature_sampler <- function(h_onehot, 
#                                       k_betw = 0, 
#                                       affinity_threshold = 1) {
#   # k_betw is the number of nodes to be removed if their betweenness levels are
#   # too low.
#   p <- length(h_onehot)
#   
#   # Each row of h_onehot consists of p one-hot vectors, each on one coordinate
#   # Ensure each element is a matrix
#   h_onehot <- lapply(h_onehot, function(h) {
#     if (is.vector(h)) matrix(h, nrow = 1) else h
#   })
#   
#   # Combine all one-hot matrices into keys to find unique rows
#   keys <- apply(do.call(cbind, h_onehot), 1, paste, collapse = "-")
#   unique_idx <- which(!duplicated(keys))
#   # Keep only unique rows in each one-hot matrix
#   h_onehot <- lapply(h_onehot, function(h) h[unique_idx, , drop = FALSE])
#   
#   # Concatenate all onehot matrices
#   H_big <- do.call(cbind, h_onehot)
#   
#   
#   # Obtain the affinity matrix
#   # Distance between (1, 0, 0, 0) and (0, 1, 0, 0) is 1
#   # Distance between (1, 0, 0, 0) and (0, 0, 1, 0) is 2
#   # Distance between (1, 0, 0, 0) and (0, 0, 0, 1) is 3
#   nonzero_fast <- lapply(1:nrow(H_big), function(i) which(H_big[i, ] != 0))
#   # L1 norm
#   # do.call(rbind, nonzero_fast) recycles entries to equalize row lengths.
#   Dmat <- as.matrix(proxy::dist(do.call(rbind, nonzero_fast), method = "Manhattan"))
#   # Affinity matrix: 1 if distance ≤ 1, else 0
#   A <- (Dmat <= affinity_threshold) * 1
#   # Remove self-loops if needed
#   diag(A) <- 0
#   
#   
#   # Build graph
#   g <- graph_from_adjacency_matrix(A, mode = "undirected")
#   # Calculate edge betweenness.
#   # High values are very important “bottleneck” edges.
#   # Low values are non-bridging edges.
#   eb <- edge_betweenness(g)
#   # Limit k_betw to the number of edges
#   k_safe <- min(k_betw, gsize(g))
#   # Only select top-k edges
#   if (k_safe > 0) {
#     top_edges <- order(eb, decreasing = TRUE)[1:k_safe]
#     # E(g) is the sequence of edges.
#     g <- delete_edges(g, E(g)[top_edges])
#   } else {
#     # No edges to remove; do nothing silently
#   }
#   
#   # In an undirected graph, weakly and strongly connected components 
#   # are identical.
#   wc <- components(g)
#   groups <- split(seq_along(wc$membership), wc$membership)
#   # Map it back to the original indices
#   newgroups <- vector("list", length(groups))
#   for (k in seq_along(groups)) {
#     newgroups[[k]] <- unique_idx[groups[[k]]]
#   }
#   
#   return(newgroups)
# }
# 
# ###
# ###
# ###
# 
# 
# cluster_sampling <- function(confidence_region, band_sampler, B = 200) {
#   
#   h_onehot <- confidence_region$h_onehot
#   group_list <- confidence_region$group_list
#   p <- length(h_onehot)
#   
#   results <- vector("list", length(group_list))
#   for (l_ in 1:length(group_list)) {
#     # Extract the corresponding one-hot vectors
#     h_onehot_output <- vector("list", p)
#     for (j in 1:p) {
#       # Collect community 1:
#       h_onehot_output[[j]] <- h_onehot[[j]][group_list[[l_]], ]
#     }
#     
#     # Preallocate matrix
#     bin_all <- attr(band_sampler, "bin_all")
#     samples_list <- vector("list", B)
#     for (b in 1:B) {
#       samples_list[[b]] <- onehot_x(h_onehot_output, bin_all)
#     }
#     samples <- do.call(rbind, samples_list)
#     results[[l_]] <- samples
#   }
#   return(results)
# }