library(vbGLMMR)
library(stanAD)

set.seed(1)
dat <- sim_GLMM_pois_data(n_subj = 3)

glmm_opt_dat <- get_data_pois_glmm(data = dat$df, formula = y ~ (1 | subj))

par <- rnorm(8)
par_scaling <- rep(1, 8)

stanAD:::test_elbo(
  par, par_scaling, glmm_opt_dat
)

elbo <- function(m, log_s, b, log_sigma, Xty, Zty, X, Z, Z2) {

  s2 <- exp(log_s) ^ 2
  sigma2 <- exp(log_sigma) ^ 2
  link <- Z %*% m + X %*% b + 0.5 * (Z2 %*% s2)
  print(link)
  elbo <- -sum(Xty * b) - sum(Zty * m) +
    sum(exp(link)) +
    0.5 * (sum(s2) + sum(m ^ 2)) * (1 / sigma2) +
    length(m) * log_sigma - sum(log_s)

  return(elbo)

}

elbo_par_fn <- function(par, Xty, Zty, X, Z, Z2) {

  elbo(
    m = par[seq(1, 6, 2)],
    log_s = par[seq(2, 6, 2)],
    b = par[7],
    log_sigma = par[8],
    Xty,
    Zty,
    X,
    Z,
    Z2
  )

}

elbo_par_fn(
  par = par,
  Xty = glmm_opt_dat$Xty,
  Zty = glmm_opt_dat$Zty,
  X = glmm_opt_dat$X,
  Z = glmm_opt_dat$Z,
  Z2 = glmm_opt_dat$Z
)

gt <- stanAD:::test_grad(
  par,
  par_scaling,
  glmm_opt_dat
)

library(numDeriv)
gg <- grad(
  func = elbo_par_fn,
  x = par,
  Xty = glmm_opt_dat$Xty,
  Zty = glmm_opt_dat$Zty,
  X = glmm_opt_dat$X,
  Z = glmm_opt_dat$Z,
  Z2 = glmm_opt_dat$Z
)






sink(file = "~/Documents/lbfgs_t2.txt")

gt <- stanAD:::test_grad(
  par,
  par_scaling,
  glmm_opt_dat
)

# I want to do a numerical gradient here that I can inspect
# this may help me in understanding if the elbo is wrong or if my grad
# calculation is off somehow

elbo <- function(m, log_s, b, log_sigma, Xty, Zty, X, Z, Z2) {

  s2 <- exp(log_s) ^ 2
  sigma2 <- exp(log_sigma) ^ 2
  link <- Z %*% m + X %*% b + 0.5 * (Z2 %*% s2)
  print(link)
  elbo <- -sum(Xty * b) - sum(Zty * m) +
    sum(exp(link)) +
    0.5 * (sum(s2) + sum(m ^ 2)) * (1 / sigma2) +
    length(m) * log_sigma - sum(log_s)

  return(elbo)

}

elbo_par_fn <- function(par, Xty, Zty, X, Z, Z2) {

  elbo(
    m = par[seq(1, 6, 2)],
    log_s = par[seq(2, 6, 2)],
    b = par[7],
    log_sigma = par[8],
    Xty,
    Zty,
    X,
    Z,
    Z2
  )

}

elbo_par_fn(
  par = par,
  Xty = glmm_opt_dat$Xty,
  Zty = glmm_opt_dat$Zty,
  X = glmm_opt_dat$X,
  Z = glmm_opt_dat$Z,
  Z2 = glmm_opt_dat$Z
)

library(numDeriv)
gg <- grad(
  func = elbo_par_fn,
  x = par,
  Xty = glmm_opt_dat$Xty,
  Zty = glmm_opt_dat$Zty,
  X = glmm_opt_dat$X,
  Z = glmm_opt_dat$Z,
  Z2 = glmm_opt_dat$Z
)


l <- c(0.463028, 1.007494, 1.007494, 0.775624, -0.472645, -0.472645, -0.472645, -0.472645,
  7.404723, 7.404723, 7.404723, 7.404723, 7.404723, 0.890933, 0.890933, 0.890933,
  2.501067, 2.501067, 2.501067, -1.049839, -1.049839, -1.049839, -1.049839, -1.049839,
  -1.049839, -1.049839, -1.049839, -1.049839, 2.953940, 4.811685, 4.811685, 4.811685,
  4.811685, 1.219272, 1.219272, 1.219272, 1.219272)

l2 <- c(1.0074944, 1.0074944, 0.7756236, -0.4726452, -0.4726452, -0.4726452, -0.4726452,
        7.4047225, 7.4047225, 7.4047225, 7.4047225, 7.4047225, 0.8909329, 0.8909329,
        0.8909329, 2.5010673, 2.5010673, 2.5010673, -1.0498387, -1.0498387, -1.0498387,
        -1.0498387, -1.0498387, -1.0498387, -1.0498387, -1.0498387, -1.0498387,
        2.9539398, 4.8116848, 4.8116848, 4.8116848, 4.8116848, 1.2192725, 1.2192725,
        1.2192725, 1.2192725, 1.2192725)
