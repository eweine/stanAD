#' Optimize Poisson GLMM with L-BFGS
#'
#' @param par initial parameters.
#' @param par_scaling scaling for parameters to ensure proper conditioning
#' @param glmm_data data for GLMM problem
#'
#' @return optimized parameters
#' @export
#'
pois_glmm_optim_lbfgs <- function(
  par,
  par_scaling,
  glmm_data
) {

  par <- par / par_scaling

  opt_out <- optim(
    par = par,
    fn = get_neg_elbo_pois_glmm,
    grad = get_grad_pois_glmm,
    method = "L-BFGS-B",
    par_scaling = par_scaling,
    X = glmm_data$X,
    vec_Z = glmm_data$ranef_Z,
    y_nz_idx = glmm_data$y_nz_idx,
    Zty = glmm_data$Zty,
    Xty = glmm_data$Xty,
    blocks_per_ranef = glmm_data$blocks_per_ranef,
    log_chol_par_per_block = glmm_data$free_cov_par_per_ranef,
    terms_per_block = glmm_data$terms_per_block,
    n = length(glmm_data$y),
    n_m_par = ncol(glmm_data$Z),
    n_log_chol_par = sum(
      glmm_data$free_cov_par_per_ranef * glmm_data$blocks_per_ranef
    ),
    n_b_par = ncol(glmm_data$X),
    total_blocks = sum(glmm_data$blocks_per_ranef)
  )

  return(opt_out$par * par_scaling)

}
