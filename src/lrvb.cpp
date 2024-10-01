#include "hvp.h"
#include <RcppEigen.h>
#include "cg.h"
#include "utils.h"
#include "hessian.h"

// Ultimately I should template this to return a matrix or a sparse matrix
// based on a user input
Eigen::MatrixXd get_lrvb_pois_glmm_mfvb(
    Eigen::VectorXd& par_vals,
    const std::vector<int>& blocks_per_ranef,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    Eigen::SparseMatrix<double>& Z,
    Eigen::SparseMatrix<double>& Z2,
    int n_ranef_par,
    int n_fixef_par
) {

  // create sparse matrices for calculating elbo
  //Eigen::SparseMatrix<double> Z;
  //Eigen::SparseMatrix<double> Z2;

  // create lambda function to get hvp
  auto get_hvp = [&par_vals, &Zty, &Xty, &X, &Z, &Z2, &blocks_per_ranef, &n_ranef_par, &n_fixef_par](const Eigen::VectorXd& v) {
    return pois_glmm_mfvb_hvp(
      par_vals,
      v,
      Zty,
      Xty,
      X,
      Z,
      Z2,
      blocks_per_ranef,
      n_ranef_par,
      n_fixef_par
    );
  };

  int total_inv_par = n_fixef_par + n_ranef_par;
  Eigen::MatrixXd H_inv(total_inv_par, total_inv_par);
  Eigen::VectorXd I_col(par_vals.size());
  Eigen::VectorXd x0(par_vals.size());
  x0.setZero();
  I_col.setZero();
  Eigen::VectorXd cg_sol;
  Eigen::VectorXd m_sol;
  Eigen::VectorXd b_sol;

  int total_par_iterated_through = 0;

  // first, find inverses of the random effects parameters
  for (int j = 0; j < n_ranef_par; j++) {

    I_col(total_par_iterated_through) = 1;
    x0(total_par_iterated_through) = -1;

    cg_sol = solve_cg(
      get_hvp,
      x0,
      I_col,
      1e-4
    );

    m_sol = cg_sol.segment(0, 2 * n_ranef_par).reshaped(2, n_ranef_par).row(0);
    b_sol = cg_sol.segment(2 * n_ranef_par, n_fixef_par);
    H_inv.col(j) << m_sol, b_sol;

    I_col(total_par_iterated_through) = 0;
    x0(total_par_iterated_through) = 0;
    total_par_iterated_through += 2;

  }

  for (int k = n_ranef_par; k < total_inv_par; k++) {

    I_col(total_par_iterated_through) = 1;
    x0(total_par_iterated_through) = -1;

    cg_sol = solve_cg(
      get_hvp,
      x0,
      I_col,
      1e-4
    );

    m_sol = cg_sol.segment(0, 2 * n_ranef_par).reshaped(2, n_ranef_par).row(0);
    b_sol = cg_sol.segment(2 * n_ranef_par, n_fixef_par);
    H_inv.col(k) << m_sol, b_sol;

    I_col(total_par_iterated_through) = 0;
    x0(total_par_iterated_through) = 0;
    total_par_iterated_through += 1;

  }

  return H_inv;

}

Eigen::MatrixXd get_lrvb_pois_glmm_mfvb_diag_precond(
    Eigen::VectorXd& par_vals,
    const std::vector<int>& blocks_per_ranef,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    Eigen::SparseMatrix<double>& Z,
    Eigen::SparseMatrix<double>& Z2,
    Eigen::VectorXd& diag_precond,
    int n_ranef_par,
    int n_fixef_par
) {

  // create sparse matrices for calculating elbo
  //Eigen::SparseMatrix<double> Z;
  //Eigen::SparseMatrix<double> Z2;

  // create lambda function to get hvp
  auto get_hvp = [&par_vals, &Zty, &Xty, &X, &Z, &Z2, &blocks_per_ranef, &n_ranef_par, &n_fixef_par](const Eigen::VectorXd& v) {
    return pois_glmm_mfvb_hvp(
      par_vals,
      v,
      Zty,
      Xty,
      X,
      Z,
      Z2,
      blocks_per_ranef,
      n_ranef_par,
      n_fixef_par
    );
  };

  int total_inv_par = n_fixef_par + n_ranef_par;
  Eigen::MatrixXd H_inv(total_inv_par, total_inv_par);
  Eigen::VectorXd I_col(par_vals.size());
  Eigen::VectorXd x0(par_vals.size());
  x0.setZero();
  I_col.setZero();
  Eigen::VectorXd cg_sol;
  Eigen::VectorXd m_sol;
  Eigen::VectorXd b_sol;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> dm_precond = diag_precond.asDiagonal();

  int total_par_iterated_through = 0;

  // first, find inverses of the random effects parameters
  for (int j = 0; j < n_ranef_par; j++) {

    I_col(total_par_iterated_through) = 1;
    // start with a good guess for the diagonal
    x0(total_par_iterated_through) = diag_precond(total_par_iterated_through);

    cg_sol = solve_cg_diag_precond(
      get_hvp,
      x0,
      I_col,
      dm_precond,
      1e-4
    );

    m_sol = cg_sol.segment(0, 2 * n_ranef_par).reshaped(2, n_ranef_par).row(0);
    b_sol = cg_sol.segment(2 * n_ranef_par, n_fixef_par);
    H_inv.col(j) << m_sol, b_sol;

    I_col(total_par_iterated_through) = 0;
    x0(total_par_iterated_through) = 0;
    total_par_iterated_through += 2;

  }

  for (int k = n_ranef_par; k < total_inv_par; k++) {

    I_col(total_par_iterated_through) = 1;
    x0(total_par_iterated_through) = -1;

    cg_sol = solve_cg(
      get_hvp,
      x0,
      I_col,
      1e-4
    );

    m_sol = cg_sol.segment(0, 2 * n_ranef_par).reshaped(2, n_ranef_par).row(0);
    b_sol = cg_sol.segment(2 * n_ranef_par, n_fixef_par);
    H_inv.col(k) << m_sol, b_sol;

    I_col(total_par_iterated_through) = 0;
    x0(total_par_iterated_through) = 0;
    total_par_iterated_through += 1;

  }

  return H_inv;

}

Eigen::VectorXd get_lrvb_approx_pois_glmm_mfvb(
    const Eigen::VectorXd& m,
    const Eigen::VectorXd& log_s,
    const Eigen::VectorXd& sigma2,
    Eigen::VectorXd& exp_link,
    const std::vector<int>& blocks_per_ranef,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    std::vector<Eigen::MatrixXd>& vec_Z,
    std::vector<std::vector<int>>& y_nz_idx,
    int n_ranef_par
) {

  Eigen::VectorXd post_approx_cov(n_ranef_par);

  int total_ranef_blocks_looped = 0;
  int par_iterated_through = 0;

  Eigen::Matrix2d block_inv;

  // loop over each random effect block
  for (int k = 0; k < blocks_per_ranef.size(); k++) {

    double sig2 = sigma2(k);

    for (
        int j = total_ranef_blocks_looped;
        j < total_ranef_blocks_looped + blocks_per_ranef[k];
        j++
      ) {

      block_inv = hess_inv_1D_pois_glmm_cpp(
        Zty(par_iterated_through),
        vec_Z[j],
        m(par_iterated_through),
        log_s(par_iterated_through),
        std::pow(std::exp(log_s(par_iterated_through)), 2),
        sig2,
        exp_link(y_nz_idx[j])
      );

      post_approx_cov(par_iterated_through) = block_inv(0, 0);
      par_iterated_through += 1;

    }

    total_ranef_blocks_looped += blocks_per_ranef[k];

  }

  return post_approx_cov;

}


Eigen::VectorXd get_lrvb_preconditioner_pois_glmm_mfvb(
    const Eigen::VectorXd& m,
    const Eigen::VectorXd& log_s,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& sigma2,
    Eigen::VectorXd& exp_link,
    const std::vector<int>& blocks_per_ranef,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    std::vector<Eigen::MatrixXd>& vec_Z,
    std::vector<std::vector<int>>& y_nz_idx,
    int n_ranef_par
) {

  Eigen::VectorXd post_approx_cov(2 * n_ranef_par + b.size());
  Eigen::VectorXd vpar_approx_cov(sigma2.size());

  int total_ranef_blocks_looped = 0;
  int par_iterated_through = 0;
  int block_inv_par_iterated_through = 0;
  int n_fixef_par = b.size();
  double ss;
  double s2;

  Eigen::Matrix2d block_inv;

  // loop over each random effect block
  for (int k = 0; k < blocks_per_ranef.size(); k++) {

    ss = 0;
    double sig2 = sigma2(k);
    double sig2_inv = 1 / sig2;
    double mk = static_cast<double>(blocks_per_ranef[k]);

    for (
        int j = total_ranef_blocks_looped;
        j < total_ranef_blocks_looped + blocks_per_ranef[k];
        j++
    ) {

      s2 = std::pow(std::exp(log_s(par_iterated_through)), 2);
      block_inv = hess_inv_1D_pois_glmm_cpp(
        Zty(par_iterated_through),
        vec_Z[j],
        m(par_iterated_through),
        log_s(par_iterated_through),
        s2,
        sig2,
        exp_link(y_nz_idx[j])
      );

      ss += std::pow(m(par_iterated_through), 2) + s2;

      post_approx_cov(block_inv_par_iterated_through) = block_inv(0, 0);
      post_approx_cov(block_inv_par_iterated_through + 1) = block_inv(1, 1);
      par_iterated_through += 1;
      block_inv_par_iterated_through += 2;

    }

    vpar_approx_cov(k) = 1.0 / (
      std::pow(sig2_inv, 3) * ss - (mk / 2.0) * std::pow(sig2_inv, 2)
    );
    total_ranef_blocks_looped += blocks_per_ranef[k];

  }

  // Now, get the diagonal of the inverse of the fixed effects
  Eigen::MatrixXd H_b = X.transpose() * exp_link.asDiagonal() * X;
  post_approx_cov.segment(block_inv_par_iterated_through, n_fixef_par) = H_b.inverse().diagonal();

  Eigen::VectorXd diag_precond(post_approx_cov.size() + vpar_approx_cov.size());
  diag_precond << post_approx_cov, vpar_approx_cov;

  return diag_precond;

}
