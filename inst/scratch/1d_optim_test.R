library(vbGLMMR)
library(stanAD)

set.seed(1)
dat <- sim_GLMM_pois_data(n_subj = 100)

total_par <- 202

glmm_opt_dat <- get_data_pois_glmm(data = dat$df, formula = y ~ (1 | subj))

opt_out <- stanAD::pois_glmm_optim_lbfgs(
  par = rnorm(n = total_par, sd = 0.01),
  par_scaling = rep(1, total_par),
  glmm_data = glmm_opt_dat
)

lm_out <- lme4::glmer(
  data = dat$df,
  formula = y ~ (1 | subj),
  family = poisson
)

opt_out2 <- stanAD::fit_pois_glmm_block_posterior(
  data = dat$df, formula = y ~ (1 | subj), num_iter = 100
)
