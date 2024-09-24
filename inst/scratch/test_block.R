library(stanAD)
set.seed(69)
n <- 100
p <- 2
Z <- MASS::mvrnorm(
  n = n,
  mu = c(0, 0),
  Sigma = matrix(
    data = c(3, 0, 0, 3),
    byrow = TRUE,
    nrow = 2
  )
)

Z2 <- Z ^ 2
m <- c(0.5, 0.5)
y <- rpois(n = n, exp(Z %*% m))

m_fit2 <- c(0, 0)
S_log_chol_fit <- rep(0, 3)
S_fit <- diag(p)

link_offset2 <- 0.5 * rowSums(Z2)

for (i in 1:100) {

  print(i)

  iter2 <- single_newton_multiD_pois_glmm_cpp_testing(
    Zty = crossprod(Z, y)[,1],
    Z = Z,
    m = m_fit2,
    S_log_chol = S_log_chol_fit,
    log_chol_diag_idx = c(2, 4),
    S = S_fit,
    Sigma_inv = solve(matrix(
      data = c(1, 0.9, 0.9, 1),
      byrow = TRUE,
      nrow = 2
    )),
    link_offset = link_offset2
  )

  m_fit2 <- iter2$m
  S_log_chol_fit <- iter2$S_log_chol
  S_fit <- iter2$S
  link_offset2 <- Z %*% m_fit2 + 0.5 * diag(Z %*% S_fit %*% t(Z))

}

ft <- fastglm::fastglmPure(
  x = Z, y = y, family = poisson()
)

library(rstan)

sout <- rstan::stan(
  file = "~/Documents/stanAD/inst/scratch/test_block.stan",
  data = list(
    N = n,
    K = p,
    X = Z,
    y = y,
    Sigma_prior = matrix(
      data = c(1, 0.9, 0.9, 1),
      byrow = TRUE,
      nrow = 2
    )
  ),
  iter = 5000
)

library(posterior)
do <- posterior::as_draws_matrix(sout)

# this really appears to be correct
