
#' Poisson GLMM with block posterior
#'
#' @param num_iter number of iterations
#' @param ... other parameters
#'
#' @return a list with fit objects
#' @export
#'
fit_pois_glmm_block_posterior <- function(
  num_iter,
  ...
) {

  parsed <- lme4::glFormula(...)
  Z <- Matrix::t(parsed$reTrms$Zt)
  Z2 <- MatrixExtra::mapSparse(Z, function(x){x^2})
  # need number of coefficients for random effect
  terms_per_block <- as.integer(lengths(parsed$reTrms$cnms))
  # need number of repetitions for each random effect
  blocks_per_ranef <- as.integer(diff(parsed$reTrms$Gp) / terms_per_block)

  free_cov_params_per_ranef <- as.integer(
    (terms_per_block * (terms_per_block+1))/2
  )

  fixef_glm <- fastglm::fastglmPure(
    x = parsed$X, y = parsed$fr$y, family = poisson(), maxit = 10
  )

  b_init <- fixef_glm$coefficients
  link_offset <- fixef_glm$linear.predictors + 0.5 * Matrix::rowSums(Z2)

  m <- rep(0, ncol(Z))
  # initialize identity posterior
  S_log_chol <- rep(0, sum(free_cov_params_per_ranef * blocks_per_ranef))
  Z_summary <- Matrix::summary(Z)
  Zty <- Matrix::crossprod(Z, parsed$fr$y)[,1]

  Z_idx <- Z
  Z_idx@x <- rep(1, length(Z_idx@x))
  n_nz_per_col <- Matrix::colSums(Z_idx)

  y <- parsed$fr$y
  X <- parsed$X
  #rm(Z, Z_idx, Z2, fixef_glm, parsed)

  fit_out <- fit_pois_glmm_block_posterior_ccd(
    Zty = Zty,
    Z_i = Z_summary$i - 1,
    Z_j = Z_summary$j - 1,
    Z_x = Z_summary$x,
    m = m,
    b = b_init,
    S_log_chol = S_log_chol,
    X = X,
    Xty = crossprod(X, y)[,1],
    n_nz_terms_per_col = n_nz_per_col,
    terms_per_block = terms_per_block,
    blocks_per_ranef = blocks_per_ranef,
    link_offset = link_offset,
    log_chol_par_per_block = free_cov_params_per_ranef,
    num_iter = num_iter
  )

  return(fit_out)

}
