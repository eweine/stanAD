#include "fpiter.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include "utils.h"


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
    const int& num_iter
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

  }

  Rcpp::List fit;
  fit["b"] = b;
  fit["m"] = m;
  fit["S"] = S;
  fit["Sigma"] = Sigma;
  return fit;

}

