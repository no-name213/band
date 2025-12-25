library(Rfast)

# Data generating model
block_gaussian <- function(n, n_blocks = 6, p = 30, rho = 0.3, sigma.report = FALSE) {
  if (p %% n_blocks != 0) stop("p must be divisible by n_blocks")
  
  block_size <- p / n_blocks
  x <- matrix(0, nrow = n, ncol = p)
  
  mu <- rep(0, block_size)
  sigma <- (1 - rho) * diag(block_size) + rho * matrix(1, nrow = block_size, ncol = block_size)
  
  # Full covariance = block diagonal matrix
  if (sigma.report) {
    Sigma_full <- Matrix::bdiag(replicate(n_blocks, sigma, simplify = FALSE))
    return(as.matrix(Sigma_full))
  }
  
  for (b in 1:n_blocks) {
    idx <- ((b - 1) * block_size + 1):(b * block_size)
    x[, idx] <- Rfast::rmvnorm(n, mu, sigma)
  }
  
  return(x)
}


rmixnorm <- function(n, p = 5, d = 0.0) {
  rho <- 0
  sigma <- outer(1:p, 1:p, FUN = function(i,j) rho^(abs(i-j)))
  mu1 <- rep(0, p)
  mu2 <- mu1 + d
  z <- rbinom(n, 1, 0.5)
  x <- matrix(0, n, length(mu1))
  idx1 <- which(z == 1)
  idx2 <- which(z == 0)
  x[idx1, ] <- Rfast::rmvnorm(length(idx1), mu1, sigma)
  x[idx2, ] <- Rfast::rmvnorm(length(idx2), mu2, sigma)
  x
}



sample_bimodal_uniform <- function(n = 1000, p = 2, d = 2.0, width = 1.5) {
  # Half from mode 1, half from mode 2
  n1 <- floor(n / 2)
  n2 <- n - n1
  
  # Mode 1: centered at 0 (uniform cube)
  x1 <- matrix(runif(n1 * p, -width, width), nrow = n1, ncol = p)
  
  # Mode 2: centered at +d in every dimension
  x2 <- matrix(runif(n2 * p, -width, width), nrow = n2, ncol = p) + d
  
  # Combine and shuffle rows randomly
  X <- rbind(x1, x2)
  X <- X[sample(1:n), , drop = FALSE]
  
  return(X)
}