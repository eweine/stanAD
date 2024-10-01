#include "fpiter.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include "utils.h"
#include "elbo.h"
#include "lrvb.h"
#include "hvp.h"


// [[Rcpp::plugins(cpp17)]]

//' @export
// [[Rcpp::export]]
Rcpp::List fit_pois_glmm_block_posterior_ccd(
    Eigen::VectorXd& m,
    Eigen::VectorXd& S_log_chol,
    Eigen::VectorXd& b,
    Eigen::VectorXd& link_offset,
    const std::vector<int>& n_nz_terms_per_col,
    const std::vector<int>& terms_per_block,
    const std::vector<int>& blocks_per_ranef,
    const std::vector<int>& log_chol_par_per_block,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    const std::vector<int>& Z_i,
    const std::vector<int>& Z_j,
    const std::vector<double>& Z_x,
    Eigen::SparseMatrix<double>& Z,
    Eigen::SparseMatrix<double>& Z2,
    double elbo_tol,
    const int& num_iter,
    const bool is_mfvb
) {

  // there's a lot to construct here
  std::vector<Eigen::MatrixXd> Sigma = initialize_Sigma(terms_per_block);

  std::vector<int> n_nz_terms = get_n_nz_terms(
    n_nz_terms_per_col,
    blocks_per_ranef,
    terms_per_block
  );

  std::vector<Eigen::MatrixXd> vec_Z = create_ranef_Z(
    n_nz_terms,
    blocks_per_ranef,
    terms_per_block,
    Z_x
  );

  std::vector<std::vector<int>> y_nz_idx = create_y_nz_idx(
    n_nz_terms, // number of nonzero terms for each ranef coefficient
    blocks_per_ranef, // number of coefficients per ranef (e.g. 50)
    terms_per_block, // number of terms in each random effect
    Z_i // values of Z
  );

  std::vector<std::vector<int>> log_chol_diag_idx_per_ranef = get_log_chol_diag_idx_per_ranef(
    terms_per_block
  );

  std::vector<Eigen::MatrixXd> S = initialize_S(
    blocks_per_ranef, terms_per_block
  );

  std::vector<double> elbo_history;
  elbo_history.reserve(num_iter + 1);

  double current_elbo = get_elbo_pois_glmm_block_posterior(
    m,
    b,
    S_log_chol,
    S, // maybe change to a vector of matrices?
    link_offset,
    Zty,
    Xty,
    blocks_per_ranef,
    log_chol_par_per_block,
    terms_per_block,
    log_chol_diag_idx_per_ranef,
    Sigma
  );
  elbo_history.push_back(current_elbo);

  double new_elbo;

  for (int i = 0; i < num_iter; i++) {

    fpiter_pois_glmm(
      m,
      S_log_chol,
      S,
      b,
      link_offset,
      X,
      Zty,
      Xty,
      blocks_per_ranef,
      log_chol_par_per_block,
      terms_per_block,
      log_chol_diag_idx_per_ranef,
      vec_Z,
      y_nz_idx,
      Sigma
    );

    new_elbo = get_elbo_pois_glmm_block_posterior(
      m,
      b,
      S_log_chol,
      S, // maybe change to a vector of matrices?
      link_offset,
      Zty,
      Xty,
      blocks_per_ranef,
      log_chol_par_per_block,
      terms_per_block,
      log_chol_diag_idx_per_ranef,
      Sigma
    );

    elbo_history.push_back(
      new_elbo
    );

    if (new_elbo - current_elbo <= elbo_tol) {

      break;

    } else {

      current_elbo = new_elbo;

    }

  }

  Rcpp::List fit;
  fit["b"] = b;
  fit["m"] = m;
  fit["S"] = S;
  fit["Sigma"] = Sigma;
  fit["elbo"] = elbo_history;

  // now, return lrvb estimates
  if (is_mfvb) {

    Rprintf("Done with optimization...\n");

    Eigen::VectorXd sigma2_inv(terms_per_block.size());
    for (int k = 0; k < terms_per_block.size(); k++) {

      sigma2_inv(k) = 1.0 / (Sigma[k](0, 0));

    }

    Eigen::VectorXd par_ranef(2 * m.size());
    par_ranef << m, S_log_chol;

    Eigen::MatrixXd par_ranef_mat = par_ranef.reshaped(m.size(), 2).transpose();

    Eigen::VectorXd par_ranef_vec = par_ranef_mat.reshaped();
    Eigen::VectorXd par_vals(2 * m.size() + b.size() + sigma2_inv.size());

    par_vals << par_ranef_vec, b, sigma2_inv;

    int n_ranef_par = m.size();
    int n_fixef_par = b.size();

    //fit["stan_math_out"] = pois_glmm_mfvb_h_test(
    //  par_vals,
    //  Zty,
    //  Xty,
    //  X,
    //  Z,
    //  Z2,
    //  blocks_per_ranef,
    //  n_ranef_par,
    //  n_fixef_par
    //);

    fit["cov"] = -get_lrvb_pois_glmm_mfvb(
      par_vals,
      blocks_per_ranef,
      Zty,
      Xty,
      X,
      Z,
      Z2,
      n_ranef_par,
      n_fixef_par
    );

  }

  return fit;

}

