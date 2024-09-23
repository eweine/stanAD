get_Sigma_from_L_tilde <- function(L_tilde) {

  diag(L_tilde) <- exp(diag(L_tilde))
  Sigma <- tcrossprod(L_tilde)
  return(Sigma)

}

fit_pois_glmm_stan <- function(
    opt_method = c("L-BFGS-B", "trust-krylov"),
    ...
) {

  #browser()
  parsed <- lme4::glFormula(...)

  opt_method = match.arg(opt_method)

  Z <- Matrix::t(parsed$reTrms$Zt)
  Z2 <- MatrixExtra::mapSparse(Z, function(x){x^2})
  # need number of coefficients for random effect
  terms_per_block <- as.integer(lengths(parsed$reTrms$cnms))
  # need number of repetitions for each random effect
  blocks_per_ranef <- as.integer(diff(parsed$reTrms$Gp) / terms_per_block)

  free_cov_params_per_ranef <- as.integer(
    (terms_per_block * (terms_per_block+1))/2
  )

  b_init <- fastglm::fastglmPure(
    x = parsed$X, y = parsed$fr$y, family = poisson(), maxit = 10
  )$coefficients

  # initialize the prior covariances at the identity
  log_sigma_chol <- mapply(function(n) ks::vech(diag(n)), terms_per_block)
  m <- rep(0, ncol(Z))
  log_s <- rep(0, ncol(Z))

  n_ranef <- ncol(Z)
  n_fixed <- ncol(parsed$X)

  par_init <- c(
    m, log_s, b_init, log_sigma_chol
  )

  if (opt_method == "L-BFGS-B") {

    tictoc::tic()
    opt_par <- optim(
      par = par_init,
      fn = get_elbo_pois_glmm_MFVB,
      gr = get_elbo_grad,
      method = "L-BFGS-B",
      control = list(fnscale = -1, maxit = 100000),
      Zty = Matrix::crossprod(Z, parsed$fr$y)[,1],
      Xty = crossprod(parsed$X, parsed$fr$y)[,1],
      X = parsed$X,
      Z = Z,
      Z2 = Z2,
      blocks_per_ranef = blocks_per_ranef,
      terms_per_block = terms_per_block,
      n_ranef_par = n_ranef,
      n_fixef_par = n_fixed
    )$par
    tictoc::toc()

  } else if (opt_method == "trust-krylov") {

    opt_fn <- function(x) {

      #browser()
      val <- -get_elbo_pois_glmm_MFVB(
        par_vals = reticulate::py_to_r(x),
        Zty = Matrix::crossprod(Z, parsed$fr$y)[,1],
        Xty = crossprod(parsed$X, parsed$fr$y)[,1],
        X = parsed$X,
        Z = Z,
        Z2 = Z2,
        blocks_per_ranef = blocks_per_ranef,
        terms_per_block = terms_per_block,
        n_ranef_par = n_ranef,
        n_fixef_par = n_fixed
      )

      return(reticulate::np_array(val))

    }

    grad_fn <- function(x) {

      #browser()
      val <- -get_elbo_grad(
        par_vals = reticulate::py_to_r(x),
        Zty = Matrix::crossprod(Z, parsed$fr$y)[,1],
        Xty = crossprod(parsed$X, parsed$fr$y)[,1],
        X = parsed$X,
        Z = Z,
        Z2 = Z2,
        blocks_per_ranef = blocks_per_ranef,
        terms_per_block = terms_per_block,
        n_ranef_par = n_ranef,
        n_fixef_par = n_fixed
      )

      return(reticulate::np_array(val))

    }

    hessp_fn <- function(x, p) {

      #browser()
      val <- -get_elbo_hvp(
        par_vals = reticulate::py_to_r(x),
        v = reticulate::py_to_r(p),
        Zty = Matrix::crossprod(Z, parsed$fr$y)[,1],
        Xty = crossprod(parsed$X, parsed$fr$y)[,1],
        X = parsed$X,
        Z = Z,
        Z2 = Z2,
        blocks_per_ranef = blocks_per_ranef,
        terms_per_block = terms_per_block,
        n_ranef_par = n_ranef,
        n_fixef_par = n_fixed
      )

      return(reticulate::np_array(val))

    }

    reticulate::use_condaenv("glmm")
    py_opt_fn <- reticulate::r_to_py(opt_fn)
    py_grad_fn <- reticulate::r_to_py(grad_fn)
    py_hessp_fn <- reticulate::r_to_py(hessp_fn)
    scipy <- reticulate::import("scipy.optimize")

    #tt <- py_opt_fn(par_init)

    tictoc::tic()
    opt_out <- scipy$minimize(
      fun = py_opt_fn,
      x0 = reticulate::r_to_py(par_init),
      jac = py_grad_fn,
      hessp = py_hessp_fn,
      method = "trust-ncg"
    )
    tictoc::toc()
    opt_par <- opt_out$x

  }

  opt_m <- opt_par[1:n_ranef]
  opt_sd <- exp(opt_par[(n_ranef + 1):(2 * n_ranef)])
  opt_b <- opt_par[(2 * n_ranef + 1):(2 * n_ranef + n_fixed)]
  opt_log_chol <- opt_par[(2 * n_ranef + n_fixed + 1):(length(par_init))]

  fit_Sigma_list <- list()

  log_chol_idx <- 1

  for (i in 1:length(parsed$reTrms$cnms)) {

    fit_Sigma_list[[names(parsed$reTrms$cnms)[i]]] <- get_Sigma_from_L_tilde(
      as.matrix(
        ks::invvech(
          opt_log_chol[
            log_chol_idx:(log_chol_idx + free_cov_params_per_ranef[i] - 1)
          ]
        )
      )
    )

  }

  return(
    list(
      m = opt_m,
      sd = opt_sd,
      b = opt_b,
      Sigma = fit_Sigma_list
    )
  )

}
