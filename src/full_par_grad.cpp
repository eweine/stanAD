#include <RcppEigen.h>
#include "grad.h"
#include "utils.h"
#include "link.h"

// [[Rcpp::export]]
Eigen::VectorXd get_grad_pois_glmm(
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
  Eigen::VectorXd grad(par.size());
  grad.setZero();

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

  int total_par_looped = 0;
  int total_ranef_blocks_looped = 0;
  int Sigma_start_idx = n_m_par + n_log_chol_par + n_b_par;
  int m_idx;
  int m_par_looped = 0;

  Eigen::VectorXd iter_link;
  Eigen::MatrixXd Sigma;
  Eigen::MatrixXd Sigma_inv;

  // loop over each random effect block
  for (int k = 0; k < terms_per_block.size(); k++) {

    m_idx = 0;

    if (terms_per_block[k] == 1) {

      Sigma = get_sigma2_from_log_sigma(
        par_scaled(Sigma_start_idx)
      );

      Eigen::VectorXd z2;
      Eigen::VectorXd m(blocks_per_ranef[k]);

      for (
          int j = total_ranef_blocks_looped;
          j < total_ranef_blocks_looped + blocks_per_ranef[k];
          j++
      ) {

        z2 = vec_Z[j].array().square().matrix();

        iter_link = link(y_nz_idx[j]);

        // this could likely be pre-computed when I get the values of S above
        iter_link -= vec_Z[j] * par_scaled(total_par_looped) +
          0.5 * z2 * S_by_block[j](0, 0);

        grad.segment(total_par_looped, 2) = single_local_block_1D_grad_pois_glmm(
          Zty(m_par_looped),
          vec_Z[j],
          z2,
          par.segment(total_par_looped, 2), // parameters in the order (m1, ls1)
          par_scaling.segment(total_par_looped, 2),
          Sigma(0, 0),
          iter_link
        );

        m(m_idx) = par_scaled(total_par_looped);
        m_idx += 1;
        m_par_looped += 1;
        total_par_looped += 2;

      }

      Eigen::VectorXd s2 = vec_S_by_ranef[k].col(0);

      grad(Sigma_start_idx) = single_var_comp_1D_grad_glmm(
        m,
        s2,
        par_scaling(Sigma_start_idx),
        par(Sigma_start_idx)
      );

      Sigma_start_idx += 1;

    } else {

      int par_per_block = log_chol_par_per_block[k] + terms_per_block[k];

      Sigma = get_Sigma_from_log_chol(
        par_scaled.segment(Sigma_start_idx, log_chol_par_per_block[k]),
        terms_per_block[k]
      );

      Sigma_inv = Sigma.inverse();

      Eigen::MatrixXd M_T(terms_per_block[k], blocks_per_ranef[k]);

      for (
          int j = total_ranef_blocks_looped;
          j < total_ranef_blocks_looped + blocks_per_ranef[k];
          j++
      ) {

        iter_link = link(y_nz_idx[j]);

        iter_link -= vec_Z[j] * par_scaled.segment(
          total_par_looped, terms_per_block[k]
        ) +
          0.5 * (vec_Z[j] * S_by_block[j]).cwiseProduct(vec_Z[j]).rowwise().sum();

        grad.segment(total_par_looped, par_per_block) = single_local_block_multiD_grad_pois_glmm(
          Zty.segment(m_par_looped, terms_per_block[k]),
          vec_Z[j],
          par.segment(total_par_looped, par_per_block), // parameters in the order (m1, ls1)
          par_scaling.segment(total_par_looped, par_per_block),
          Sigma_inv,
          iter_link,
          terms_per_block[k]
        );

        M_T.col(m_idx) = par_scaled.segment(total_par_looped, terms_per_block[k]);
        m_idx += 1;
        m_par_looped += terms_per_block[k];
        total_par_looped += par_per_block;

      }

      M_T.transposeInPlace();

      grad.segment(Sigma_start_idx, log_chol_par_per_block[k]) = single_var_comp_multiD_grad_glmm(
        M_T,
        vec_S_by_ranef[k],
        par_scaling.segment(Sigma_start_idx, log_chol_par_per_block[k]),
        par.segment(Sigma_start_idx, log_chol_par_per_block[k]),
        terms_per_block[k]
      );

      Sigma_start_idx += log_chol_par_per_block[k];

    }

    total_ranef_blocks_looped += blocks_per_ranef[k];

  }

  // need to subtract link here
  link -= X * par_scaled.segment(n_m_par + n_log_chol_par, n_b_par);
  grad.segment(n_m_par + n_log_chol_par, n_b_par) = fixef_grad_pois_glmm(
    Xty,
    X,
    par.segment(n_m_par + n_log_chol_par, n_b_par),
    par_scaling.segment(n_m_par + n_log_chol_par, n_b_par),
    link
  );

  return grad;

}
