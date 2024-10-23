# here, I want to make sure that the code works when there
# are multiple separate random effects that are crossed
library(stanAD)
library(vbGLMMR)

set.seed(1)
dat <- sim_GLMM_pois_data_block_cov(
  n_subj = 100,
  b_0 = 0.5,
  b_1 = 0.1,
  ranef_sigma_int = 0.9,
  ranef_sigma_slope = 0.4,
  ranef_rho = 0.1
)

dat$df$sex <- as.factor(sample(
  c("M", "F", "NA", "NB", "FTM", "MTF"),
  size = nrow(dat$df),
  replace = TRUE
  ))

total_par <- 100 * 5 + 6 * 2 + 3 + 2 + 1

#sink(file = "~/Documents/multiD_test3.txt")

glmm_opt_dat <- get_data_pois_glmm(
  data = dat$df,
  formula = y ~ x + (1 + x | subj) + (1 | sex)
  )

par_init <- rnorm(n = total_par, sd = 0.01)

print("getting elbo...")

et <- stanAD:::test_elbo(
  par = par_init,
  par_scaling = rep(1, total_par),
  glmm_data = glmm_opt_dat
)

opt_out <- stanAD::pois_glmm_optim_lbfgs(
  par = rnorm(n = total_par, sd = 0.01),
  par_scaling = rep(1, total_par),
  glmm_data = glmm_opt_dat
)

opt_out2 <- stanAD::fit_pois_glmm_block_posterior(
  data = dat$df, formula = y ~ x + (1 + x | subj) + (1 | sex), num_iter = 500
)
