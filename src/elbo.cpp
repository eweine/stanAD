#include <RcppEigen.h>
#include "elbo.h"

double get_elbo_pois_glmm_block_posterior(
    Eigen::VectorXd& m,
    Eigen::VectorXd& b,
    Eigen::VectorXd& S_log_chol,
    std::vector<Eigen::MatrixXd>& S, // maybe change to a vector of matrices?
    Eigen::VectorXd& link,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const std::vector<int>& blocks_per_ranef,
    const std::vector<int>& log_chol_par_per_block,
    const std::vector<int>& terms_per_block,
    const std::vector<std::vector<int>>& log_chol_diag_idx_per_ranef,
    std::vector<Eigen::MatrixXd>& Sigma
) {

  double elbo = Zty.dot(m) + Xty.dot(b) - link.array().exp().sum();

  // Now, need to go through and calculate remaining values
  int m_par_iterated_through = 0;
  int total_ranef_blocks_looped = 0;
  int log_chol_par_iterated_through = 0;

  Eigen::MatrixXd Sigma_inv;
  Eigen::MatrixXd M;
  std::vector<int> log_chol_diag_idx;
  int block_terms;
  int log_chol_par;

  // loop over each random effect block
  for (int k = 0; k < terms_per_block.size(); k++) {

    block_terms = terms_per_block[k];
    log_chol_par = log_chol_par_per_block[k];

    elbo -= 0.5 * static_cast<double>(blocks_per_ranef[k]) * std::log(
      Sigma[k].determinant()
    );

    Sigma_inv = Sigma[k].inverse();

    M = m.segment(
      m_par_iterated_through, blocks_per_ranef[k] * block_terms
    ).reshaped(
        block_terms, blocks_per_ranef[k]
    ).transpose();

    elbo -= 0.5 * ((M * Sigma_inv).array() * M.array()).sum();
    log_chol_diag_idx = log_chol_diag_idx_per_ranef[k];
    std::for_each(
      std::begin(log_chol_diag_idx),
      std::end(log_chol_diag_idx),
      [&block_terms, &log_chol_par_iterated_through](int& x) {
        x += log_chol_par_iterated_through - block_terms;
      }
    );

    for (
        int j = total_ranef_blocks_looped;
        j < total_ranef_blocks_looped + blocks_per_ranef[k];
        j++
      ) {

      elbo -= 0.5 * (S[j] * Sigma_inv).diagonal().sum();
      elbo += S_log_chol(log_chol_diag_idx).sum();
      // update indices for the log cholesky
      std::for_each(
        std::begin(log_chol_diag_idx),
        std::end(log_chol_diag_idx),
        [&log_chol_par](int& x) {
          x += log_chol_par;
          }
      );

    }

    total_ranef_blocks_looped += blocks_per_ranef[k];
    m_par_iterated_through += blocks_per_ranef[k] * block_terms;
    log_chol_par_iterated_through += blocks_per_ranef[k] * log_chol_par;

  }

  return elbo;

}
