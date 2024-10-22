# let's first just see if the gradients agree at all between the numerical
# and automatic methods...

library(vbGLMMR)
library(stanAD)

set.seed(10000)
dat <- sim_GLMM_pois_data_block_cov(
  n_subj = 2,
  b_0 = 0,
  b_1 = 1,
  ranef_sigma_int = 0.5,
  ranef_sigma_slope = 0.25,
  ranef_rho = 0.05
)

glmm_opt_dat <- get_data_pois_glmm(data = dat$df, formula = y ~ x + (1 + x | subj))

par <- rnorm(15, sd = 0.1)
par_scaling <- rep(1, 15)

#sink(file = "~/Documents/md_grad6.txt")

get_elbo_2D <- function(M_list, S2_list, b, Sigma, X, y, vec_Z, y_nz_idx) {

  det_S_term <- 0
  mSm_term <- 0
  tr_term <- 0
  det_Sigma_term <- 0

  elbo <- 0
  Sigma_inv <- solve(Sigma)

  link <- X %*% b
  lin_link <- X %*% b
  det_Sig <- as.numeric(determinant(Sigma, logarithm = TRUE)$modulus)

  for (i in 1:2) {

    print("link = ")
    print(link)

    link[y_nz_idx[[i]] + 1] <- link[y_nz_idx[[i]] + 1] + vec_Z[[i]] %*% M_list[[i]] +
      0.5 * diag(vec_Z[[i]] %*% S2_list[[i]] %*% t(vec_Z[[i]]))

    lin_link[y_nz_idx[[i]] + 1] <- lin_link[y_nz_idx[[i]] + 1] + vec_Z[[i]] %*% M_list[[i]]

    print("S = ")
    print(S2_list[[i]])
    print("m = ")
    print(M_list[[i]])

    print("diag_product = ")
    print(diag(vec_Z[[i]] %*% S2_list[[i]] %*% t(vec_Z[[i]])))

    det_S_term <- det_S_term - 0.5 * as.numeric(determinant(S2_list[[i]], logarithm = TRUE)$modulus)
    mSm_term <- mSm_term +  0.5 * as.numeric(t(M_list[[i]]) %*% Sigma_inv %*% M_list[[i]])
    tr_term <- tr_term + 0.5 * sum(diag(S2_list[[i]] %*% Sigma_inv))
    det_Sigma_term <- det_Sigma_term + 0.5 * det_Sig

    elbo <- elbo - 0.5 * as.numeric(determinant(S2_list[[i]], logarithm = TRUE)$modulus) +
      0.5 * det_Sig + 0.5 * as.numeric(t(M_list[[i]]) %*% Sigma_inv %*% M_list[[i]]) +
      0.5 * sum(diag(S2_list[[i]] %*% Sigma_inv))

  }

  print("link = ")
  print(link)

  print("det_S_term = ")
  print(det_S_term)

  print("mSm_term = ")
  print(mSm_term)

  print("tr_term = ")
  print(tr_term)

  print("det_Sigma_term = ")
  print(det_Sigma_term)

  print("lin_term = ")
  print(- sum(y * lin_link))

  print("exp_term = ")
  print(sum(exp(link)))

  elbo <- elbo - sum(y * lin_link) + sum(exp(link))
  return(elbo)

}

get_Sigma_from_log_chol_2D <- function(log_chol) {

  L <- matrix(data = 0, nrow = 2, ncol = 2)
  L[lower.tri(L, diag = TRUE)] <- log_chol
  diag(L) <- exp(diag(L))
  Sigma <- L %*% t(L)

  return(Sigma)

}

get_elbo_2D_par_fn <- function(par, X, y, vec_Z, y_nz_idx) {

  M_list <- list()
  S2_list <- list()

  M_list[[1]] <- par[1:2]
  S2_list[[1]] <- get_Sigma_from_log_chol_2D(par[3:5])

  M_list[[2]] <- par[6:7]
  S2_list[[2]] <- get_Sigma_from_log_chol_2D(par[8:10])

  b <- par[11:12]
  Sigma <- get_Sigma_from_log_chol_2D(par[13:15])
  print(Sigma)

  get_elbo_2D(M_list, S2_list, b, Sigma, X, y, vec_Z, y_nz_idx)

}


get_elbo_2D_par_fn(
  par,
  glmm_opt_dat$X,
  glmm_opt_dat$y,
  glmm_opt_dat$ranef_Z,
  glmm_opt_dat$y_nz_idx
  )


gt <- stanAD:::test_grad(
  par,
  par_scaling,
  glmm_opt_dat
)

et <- stanAD:::test_elbo(
  par,
  par_scaling,
  glmm_opt_dat
)

library(numDeriv)
g <- grad(
  func = get_elbo_2D_par_fn,
  x = par,
  X = glmm_opt_dat$X,
  y = glmm_opt_dat$y,
  vec_Z = glmm_opt_dat$ranef_Z,
  y_nz_idx = glmm_opt_dat$y_nz_idx
)

# I think at first I should figure out if the elbo is right
# once I'm there I can move on
