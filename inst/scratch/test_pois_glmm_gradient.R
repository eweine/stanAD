# first, I will test the 1D case
n <- 10
z <- rnorm(n)

m <- 1.0
log_s <- 1.0
lambda <- exp(z * m)
y <- rpois(n, lambda)

sigma2 <- 1

elbo_1d_reg <- function(m, log_s, link, z, y, sigma2) {

  s2 <- (exp(log_s) ^ 2)
  -sum(y * (z * m)) + sum(exp(z * m + 0.5 * (z^2) * s2 + link)) -log_s +
    0.5 * (m^2) * (1 / sigma2) + 0.5 * (s2 / sigma2)

}

elbo_1d_reg_par_fn <- function(par, par_scaling, link, z, y, sigma2) {

  par <- par * par_scaling

  elbo_1d_reg(
    m = par[1], log_s = par[2], link = link, z = z, y = y, sigma2 = sigma2
  )

}
library(tictoc)
library(numDeriv)

tic()
grad(
  func = elbo_1d_reg_par_fn,
  x = c(m, log_s),
  par_scaling = rep(2, 2),
  link = rep(0.1, n),
  z = z,
  y = y,
  sigma2 = sigma2
)
toc()

tic()
stanAD:::single_local_block_1D_grad_pois_glmm(
  Zty = sum(z * y),
  z = z,
  z2 = (z^2),
  par_vals = c(m, log_s),
  par_scaling = c(2, 2),
  sigma2 = 1,
  link = rep(0.1, n)
)
toc()


stanAD:::single_local_block_multiD_grad_pois_glmm(
  Zty = sum(z * y),
  Z = as.matrix(z),
  par_vals = c(m, log_s),
  par_scaling = c(2, 2),
  Sigma_inv = matrix(data = 1/sigma2, nrow = 1, ncol = 1),
  link = rep(0.1, n),
  n_m_par = 1
)


var_comp_1d <- function(par, scaling, mb, v2) {

  par <- par * scaling
  sigma2 <- exp(par) ^ 2
  ((0.5 * sum(mb^2)) / sigma2) + ((0.5 * sum(v2)) / sigma2) + length(mb) * par

}

p <- 100
m <- rnorm(p)
s2 <- runif(p)

grad(
  func = var_comp_1d,
  x = 0,
  scaling = 2,
  mb = m,
  v2 = s2
)


stanAD:::single_var_comp_1D_grad_glmm(
  m = m,
  s2 = s2,
  par_scaling = 2,
  log_sigma = 0
)

var_comp_multiD <- function(log_sigma_par, scaling, M_list, S2_list) {

  log_sigma_par <- log_sigma_par * scaling
  L <- matrix(data = 0, nrow = length(M_list[[1]]), ncol = length(M_list[[1]]))
  L[lower.tri(L, diag = TRUE)] <- log_sigma_par
  diag(L) <- exp(diag(L))
  Sigma <- L %*% t(L)

  p <- length(M_list)

  elbo <- 0.5 * p * as.numeric(determinant(
    Sigma, logarithm = TRUE
  )$modulus)

  Sigma_inv <- solve(Sigma)

  for (j in 1:p) {

    elbo <- elbo + 0.5 * (
      sum(diag(S2_list[[j]] %*% Sigma_inv)) + as.numeric(
        t(M_list[[j]]) %*% Sigma_inv %*% M_list[[j]]
      )
    )

  }

  return(elbo)

}

p <- 100
M <- matrix(nrow = p, ncol = 2)

for (j in 1:p) {

  M[j, ] <- rnorm(2)

}

M_list <- split(M, row(M))

S2_list <- list()
S2 <- matrix(nrow = p, ncol = 4)

for (j in 1:p) {

  rho <- runif(1)
  S2_list[[j]] <- matrix(
    data = c(1, rho, rho, 1),
    nrow = 2,
    byrow = TRUE
  )

  S2[j, ] <- as.vector(S2_list[[j]])

}

log_sigma_par <- c(0, 0.1, 0)

tic()
grad(
  func = var_comp_multiD,
  x = log_sigma_par,
  scaling = rep(1, 3),
  M_list = M_list,
  S2_list = S2_list
)
toc()

tic()
stanAD:::single_var_comp_multiD_grad_glmm(
  M = M,
  S = S2,
  par_scaling = rep(1, 3),
  Sigma_log_chol = log_sigma_par,
  Sigma_d = 2
)
toc()

elbo_multiD_reg <- function(m, S, link, Z, y, Sigma_inv) {

  -sum(y * (Z %*% m)) +
    sum(exp(Z %*% m + 0.5 * diag(Z %*% S %*% t(Z)) + link)) +
    0.5 * as.numeric(t(m) %*% Sigma_inv %*% m) +
    0.5 * sum(diag(S %*% Sigma_inv)) -
    0.5 * as.numeric(determinant(S, logarithm = TRUE)$modulus)

}

elbo_multiD_reg_par_fn <- function(par, par_scaling, link, Z, y, Sigma_inv) {

  par <- par * par_scaling
  m <- par[1:ncol(Z)]
  log_S_chol <- par[(ncol(Z) + 1):length(par)]
  L <- matrix(data = 0, nrow = length(m), ncol = length(m))
  L[lower.tri(L, diag = TRUE)] <- log_S_chol
  diag(L) <- exp(diag(L))
  S <- L %*% t(L)

  elbo_multiD_reg(
    m = m, S = S, link = link, Z = Z, y = y, Sigma_inv = Sigma_inv
  )

}

n <- 10
# finally, I just want to test the multi-dimensional version
Z <- matrix(
  data = rnorm(n * 2), nrow = n, ncol = 2
)

m <- rnorm(2)
log_S_chol <- c(0, 0.1, 0)
Sigma_inv <- solve(
  matrix(
    data = c(1, 0.1, 0.1, 1),
    nrow = 2,
    byrow = TRUE
  )
)

y <- rpois(n, 1)

tic()
grad(
  func = elbo_multiD_reg_par_fn,
  x = c(m, log_S_chol),
  par_scaling = rep(1, 5),
  link = rep(0, n),
  Z = Z,
  y = y,
  Sigma_inv = Sigma_inv
)
toc()

tic()
stanAD:::single_local_block_multiD_grad_pois_glmm(
  Zty = (t(Z) %*% y)[,1],
  Z = Z,
  par_vals = c(m, log_S_chol),
  par_scaling = rep(1, 5),
  Sigma_inv = Sigma_inv,
  link = rep(0, n),
  n_m_par = 2
)
toc()
