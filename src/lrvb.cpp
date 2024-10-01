#include "hvp.h"
#include <RcppEigen.h>
#include "cg.h"
#include "utils.h"

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
