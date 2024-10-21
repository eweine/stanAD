#include <RcppEigen.h>
#include "utils.h"
#include "link.h"

// [[Rcpp::export]]
double get_neg_elbo_pois_glmm(
    Eigen::VectorXd& par,
    Eigen::VectorXd& par_scaling,
    Eigen::MatrixXd& X,
    std::vector<Eigen::MatrixXd>& vec_Z,
    std::vector<std::vector<int>>& y_nz_idx,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const std::vector<int>& blocks_per_ranef,
    const std::vector<int>& log_chol_par_per_block,
    const std::vector<int>& terms_per_block,
    int& n, // total number of datapoints
    int& n_m_par,
    int& n_log_chol_par,
    int& n_b_par,
    int& total_blocks
) {

  Eigen::VectorXd par_scaled = par.cwiseProduct(par_scaling);

  Eigen::VectorXd link(n);
  link.setZero();
  std::vector<Eigen::MatrixXd> S_by_block;
  S_by_block.reserve(total_blocks);

  int fixef_start = n_m_par + n_log_chol_par;

  std::vector<Eigen::MatrixXd> vec_S_by_ranef = get_link_pois_glmm(
    par_scaled,
    X,
    vec_Z,
    y_nz_idx,
    blocks_per_ranef,
    log_chol_par_per_block,
    terms_per_block,
    link,
    S_by_block,
    fixef_start,
    n_b_par
  );

  Rprintf("Printing link...\n");
  printVector(link);

  double neg_elbo = link.array().exp().sum();

  int total_par_looped = 0;
  int total_ranef_blocks_looped = 0;
  int Sigma_start_idx = fixef_start + n_b_par;
  int m_par_looped = 0;

  Eigen::MatrixXd Sigma;
  Eigen::MatrixXd L;
  Eigen::MatrixXd Sigma_inv;

  // loop over each random effect block
  for (int k = 0; k < terms_per_block.size(); k++) {

    if (terms_per_block[k] == 1) {

      Sigma = get_sigma2_from_log_sigma(
        par_scaled(Sigma_start_idx)
      );

      for (
          int j = total_ranef_blocks_looped;
          j < total_ranef_blocks_looped + blocks_per_ranef[k];
          j++
      ) {

          // here, need to get other terms
          neg_elbo += -Zty(m_par_looped) * par_scaled(total_par_looped) +
            0.5 * (1 / Sigma(0, 0)) * (std::pow(par_scaled(total_par_looped), 2) + S_by_block[j](0, 0)) -
            par_scaled(total_par_looped + 1);

          m_par_looped += 1;
          total_par_looped += 2;

      }

      neg_elbo += static_cast<double>(blocks_per_ranef[k]) * par_scaled(Sigma_start_idx);
      Sigma_start_idx += 1;

    } else {

      int par_per_block = log_chol_par_per_block[k] + terms_per_block[k];

      L = get_L_from_log_chol(
        par_scaled.segment(Sigma_start_idx, log_chol_par_per_block[k]),
        log_chol_par_per_block[k]
      );

      neg_elbo += static_cast<double>(blocks_per_ranef[k]) * L.diagonal().sum();
      L.diagonal() = L.diagonal().array().exp();
      Sigma = L * L.transpose();
      Sigma_inv = Sigma.inverse();

      for (
          int j = total_ranef_blocks_looped;
          j < total_ranef_blocks_looped + blocks_per_ranef[k];
          j++
      ) {

        neg_elbo += -Zty.segment(m_par_looped, terms_per_block[k]).dot(
          par_scaled.segment(total_par_looped, terms_per_block[k])
        ) + 0.5 * Sigma_inv.cwiseProduct(S_by_block[j]).sum() +
          0.5 * par_scaled.segment(total_par_looped, terms_per_block[k]).dot(
            Sigma_inv * par_scaled.segment(total_par_looped, terms_per_block[k])
          ) + get_det_from_log_chol(
              par_scaled.segment(
                total_par_looped + terms_per_block[k],
                log_chol_par_per_block[k]
              ),
              log_chol_par_per_block[k]
          );

        m_par_looped += terms_per_block[k];
        total_par_looped += par_per_block;

      }

      Sigma_start_idx += log_chol_par_per_block[k];

    }

    total_ranef_blocks_looped += blocks_per_ranef[k];

  }

  neg_elbo -= Xty.dot(par_scaled.segment(fixef_start, n_b_par));

  return neg_elbo;

}
