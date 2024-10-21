#' Get Data for Poisson GLMM
#'
#' @param ... parameters to be passed to lme4
#'
#' @return list
#' @export
#'
get_data_pois_glmm <- function(...) {

  parsed <- lme4::glFormula(...)
  Z <- Matrix::t(parsed$reTrms$Zt)

  # need number of coefficients for random effect
  terms_per_block <- as.integer(lengths(parsed$reTrms$cnms))
  # need number of repetitions for each random effect
  blocks_per_ranef <- as.integer(diff(parsed$reTrms$Gp) / terms_per_block)

  Z_idx <- Z
  Z_idx@x <- rep(1, length(Z_idx@x))
  n_nz_per_col <- Matrix::colSums(Z_idx)

  n_nz_terms <- get_n_nz_terms(
    n_nz_per_col,
    blocks_per_ranef,
    terms_per_block
  )

  Z_summary <- Matrix::summary(Z)

  ranef_Z <- create_ranef_Z(
    n_nz_terms,
    blocks_per_ranef,
    terms_per_block,
    Z_summary$x
  )

  y_nz_idx <- create_y_nz_idx(
    n_nz_terms,
    blocks_per_ranef,
    terms_per_block,
    Z_summary$i - 1
  )

  free_cov_par_per_ranef <- terms_per_block * (terms_per_block + 1) * 0.5

  return(
    list(
      terms_per_block = terms_per_block,
      blocks_per_ranef = blocks_per_ranef,
      n_nz_terms = n_nz_terms,
      ranef_Z = ranef_Z,
      y_nz_idx = y_nz_idx,
      X = parsed$X,
      y = parsed$fr$y,
      Zty = Matrix::crossprod(Z, parsed$fr$y)[,1],
      Xty = crossprod(parsed$X, parsed$fr$y)[,1],
      free_cov_par_per_ranef = free_cov_par_per_ranef,
      Z = Z
    )
  )

}

init_id_log_chol <- function(n) {

  rep(0, as.integer(0.5 * n * (n + 1)))

}

init_std_norm <- function(n) {

  c(rep(0, as.integer(n + (n * (n + 1) / 2))))

}


init_params <- function(blocks_per_ranef, terms_per_block, n_fixef_par) {

  free_cov_par_per_ranef <- terms_per_block * (terms_per_block + 1) * 0.5
  return(
    rep(
      0,
      n_fixef_par +
        sum(blocks_per_ranef * (terms_per_block + free_cov_par_per_ranef)) +
        sum(free_cov_par_per_ranef)
    )
  )

}
