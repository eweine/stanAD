#include <RcppEigen.h>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
double get_elbo_pois_glmm_MFVB(
    const Eigen::VectorXd& par_vals,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    const Eigen::SparseMatrix<double>& Z,
    const Eigen::SparseMatrix<double>& Z2,
    const std::vector<int>& blocks_per_ranef,
    const std::vector<int>& terms_per_block,
    int& n_ranef_par,
    int& n_fixef_par
) {

  int n_log_chol_par = par_vals.size() - (2*n_ranef_par + n_fixef_par);
  Eigen::VectorXd m = par_vals.segment(0, n_ranef_par);
  Eigen::VectorXd log_s = par_vals.segment(n_ranef_par, n_ranef_par);
  Eigen::VectorXd b = par_vals.segment(2*n_ranef_par, n_fixef_par);
  Eigen::VectorXd sigma_log_chol = par_vals.segment(2*n_ranef_par + n_fixef_par, n_log_chol_par);

  // Calculate s2 = (exp(log_s))^2
  Eigen::VectorXd s2 = log_s.array().exp().square();

  // Calculate the link vector: link = Z * m + 0.5 * Z2 * s2
  Eigen::VectorXd link = Z * m + 0.5 * Z2 * s2 + X * b;

  double elbo = 0;
  elbo += Zty.dot(m) +
    Xty.dot(b) -
    link.array().exp().sum() +
    log_s.sum();


  int cols_iterated_through = 0;
  int log_chol_par_iterated_through = 0;

  Eigen::MatrixXd Sigma;
  Eigen::MatrixXd Sigma_inv;

  Eigen::VectorXd m_block;
  Eigen::VectorXd s_block;
  Eigen::MatrixXd M;
  //
  // // loop over each random effect block
  for (int k = 0; k < terms_per_block.size(); k++) {

    Eigen::MatrixXd L_tilde(terms_per_block[k], terms_per_block[k]);
    L_tilde.setZero();

    // Fill the lower triangular matrix L with the log-Cholesky parameterization
    // in column major order
    for (int j = 0; j < terms_per_block[k]; ++j) {
      for (int i = j; i < terms_per_block[k]; ++i) {
        L_tilde(i, j) = sigma_log_chol(log_chol_par_iterated_through);
        log_chol_par_iterated_through++;
      }
    }

    elbo -= static_cast<double>(blocks_per_ranef[k]) * L_tilde.diagonal().sum();
    L_tilde.diagonal() = L_tilde.diagonal().array().exp();
    Sigma = L_tilde * L_tilde.transpose();
    Sigma_inv = Sigma.inverse();

    s_block = s2.segment(
      cols_iterated_through, blocks_per_ranef[k] * terms_per_block[k]
    );

    m_block = m.segment(
      cols_iterated_through, blocks_per_ranef[k] * terms_per_block[k]
    );

    M = m_block.reshaped(
      terms_per_block[k], blocks_per_ranef[k]
    ).transpose();

    elbo -= 0.5 * ((M * Sigma_inv).array() * M.array()).sum();

    elbo -= 0.5 * s_block.reshaped(
       terms_per_block[k],
       blocks_per_ranef[k]
    ).rowwise().sum().dot(Sigma_inv.diagonal());

    cols_iterated_through += blocks_per_ranef[k] * terms_per_block[k];

  }

  return elbo;
}

