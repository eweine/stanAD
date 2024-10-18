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

  double neg_elbo = link.array().exp().sum();

  int total_par_looped = 0;
  int total_ranef_blocks_looped = 0;
  int Sigma_start_idx = n_m_par + n_log_chol_par + n_b_par;
  int m_idx;
  int m_par_looped = 0;
  double par_sum = 0;
  double det_scaling = 0;

  Eigen::VectorXd iter_link;
  Eigen::MatrixXd Sigma;
  Eigen::MatrixXd L;
  Eigen::MatrixXd Sigma_inv;

  // loop over each random effect block
  for (int k = 0; k < terms_per_block.size(); k++) {

    m_idx = 0;

    if (terms_per_block[k] == 1) {

      Sigma = get_sigma2_from_log_sigma(
        par_scaled(Sigma_start_idx)
      );

      Eigen::VectorXd z2;
      Eigen::VectorXd m2(blocks_per_ranef[k]);

      for (
          int j = total_ranef_blocks_looped;
          j < total_ranef_blocks_looped + blocks_per_ranef[k];
          j++
      ) {

          m2(m_idx) = std::pow(par_scaled(total_par_looped), 2);
          // here, need to get other terms
          neg_elbo += -Zty(m_par_looped) * par_scaled(total_par_looped) +
            0.5 * (1 / Sigma(0, 0)) * (m2(m_idx) + vec_S_by_ranef[k](0, m_idx)) -
            par_scaled(total_par_looped + 1);

          m_idx += 1;
          m_par_looped += 1;
          total_par_looped += 2;

      }

      Eigen::VectorXd s2 = vec_S_by_ranef[k].row(0);
      par_sum = 0.5 * (
        m2.sum() + s2.sum()
      );

      det_scaling = static_cast<double>(m2.size());

      neg_elbo += det_scaling * par_scaled(Sigma_start_idx) +
        (par_sum / Sigma(0, 0));

      Sigma_start_idx += 1;

    } else {

      int par_per_block = log_chol_par_per_block[k] + terms_per_block[k];

      L = get_L_from_log_chol(
        par_scaled.segment(Sigma_start_idx, log_chol_par_per_block[k]),
        log_chol_par_per_block[k]
      );

      neg_elbo += blocks_per_ranef[k] * L.diagonal().sum();
      L.diagonal() = L.diagonal().array().exp();
      Sigma = L * L.transpose();
      Sigma_inv = Sigma.inverse();

      Eigen::MatrixXd M_T(terms_per_block[k], blocks_per_ranef[k]);

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

        M_T.col(m_idx) = par.segment(total_par_looped, terms_per_block[k]);
        m_idx += 1;
        m_par_looped += terms_per_block[k];
        total_par_looped += par_per_block;

      }

      M_T.transposeInPlace();

      neg_elbo += 0.5 * (
        M_T.cwiseProduct(M_T * Sigma_inv).sum() +
          (vec_S_by_ranef[k] * Sigma_inv.reshaped()).sum()
      );

      Sigma_start_idx += log_chol_par_per_block[k];

    }

    total_ranef_blocks_looped += blocks_per_ranef[k];

  }

  neg_elbo -= Xty.dot(par_scaled.segment(n_m_par + n_log_chol_par, n_b_par));

  return neg_elbo;

}
