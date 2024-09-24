#include <RcppEigen.h>

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
    std::vector<Eigen::MatrixXd>& vec_Z,
    std::vector<std::vector<int>>& y_nz_idx,
    std::vector<Eigen::MatrixXd>& Sigma
) {

  double elbo = Zty.dot(m) + Xty.dot(b) - link.array().exp().sum();

  // Now, need to go through and calculate remaining values
  int m_par_iterated_through = 0;
  int log_chol_par_iterated_through = 0;
  int total_ranef_blocks_looped = 0;

  Eigen::MatrixXd Sigma_inv;
  Eigen::MatrixXd M;

  // loop over each random effect block
  for (int k = 0; k < terms_per_block.size(); k++) {

    elbo -= 0.5 * static_cast<double>(blocks_per_ranef[k]) * std::log(
      Sigma[k].determinant()
    );

    Sigma_inv = Sigma[k].inverse();

    M = m.segment(
      m_par_iterated_through, blocks_per_ranef[k] * terms_per_block[k]
    ).reshaped(
        terms_per_block[k], blocks_per_ranef[k]
    ).transpose();

    elbo -= 0.5 * ((M * Sigma_inv).array() * M.array()).sum();

    // Compute other terms here
    // RESTART HERE


    m_par_iterated_through += blocks_per_ranef[k] * terms_per_block[k];

  }

  return elbo;

}
