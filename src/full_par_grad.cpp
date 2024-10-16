#include <RcppEigen.h>
#include "grad.h"

Eigen::MatrixXd get_S(Eigen::VectorXd log_chol_par, int terms_per_block) {

  Eigen::MatrixXd L(terms_per_block, terms_per_block);
  L.setZero();

  int index_s = 0;

  for (int k = 0; k < terms_per_block; ++k) {
    for (int l = k; l < terms_per_block; ++l) {

      L(l, k) = log_chol_par(index_s);

      index_s++;
    }
  }

  L.diagonal() = L.diagonal().array().exp();
  return L * L.transpose();

}

Eigen::MatrixXd get_S_1d(double log_sigma) {

  Eigen::Matrix<double, 1, 1> S;
  S(0, 0) = std::pow(std::exp(log_sigma), 2);

  return S;

}

// calculates link function and S for each block
std::vector<Eigen::MatrixXd> get_link(
  Eigen::VectorXd& scaled_par,
  std::vector<Eigen::MatrixXd>& vec_Z,
  std::vector<std::vector<int>>& y_nz_idx,
  const std::vector<int>& blocks_per_ranef,
  const std::vector<int>& log_chol_par_per_block,
  const std::vector<int>& terms_per_block,
  Eigen::VectorXd& link,
  std::vector<Eigen::MatrixXd>& S_by_block
) {

  std::vector<Eigen::MatrixXd> vec_S_by_ranef;
  vec_S_by_ranef.reserve(terms_per_block.size());
  int total_par_looped = 0;
  int total_ranef_blocks_looped = 0;

  for (int k = 0; k < terms_per_block.size(); k++) {

    Eigen::MatrixXd S_T(
        terms_per_block[k] * terms_per_block[k],
        blocks_per_ranef[k]
    );
    S_T.setZero();

    if (terms_per_block[k] == 1) {

      Eigen::Matrix<double, 1, 1> S_j;
      S_j.setZero();

      for (
          int j = total_ranef_blocks_looped;
          j < total_ranef_blocks_looped + blocks_per_ranef[k];
          j++
      ) {

        S_j = get_S_1d(scaled_par(total_par_looped + 1));
        S_by_block.push_back(S_j);
        S_T(0, j) = S_j(0, 0);

        link(y_nz_idx[j]) += vec_Z[j] * scaled_par(total_par_looped) +
          0.5 * vec_Z[j].array().square().matrix() * S_j(0, 0);

        total_par_looped += 2;

      }

    } else {

      Eigen::MatrixXd S_j(terms_per_block[k], terms_per_block[k]);
      S_j.setZero();

      for (
          int j = total_ranef_blocks_looped;
          j < total_ranef_blocks_looped + blocks_per_ranef[k];
          j++
      ) {

        S_j = get_S(
          scaled_par.segment(
            total_par_looped + terms_per_block[k],
            log_chol_par_per_block[k]
          ),
          terms_per_block[k]
        );

        S_T.col(j) = S_j.reshaped();

        link(y_nz_idx[j]) += vec_Z[j] * scaled_par.segment(
          total_par_looped, terms_per_block[k]
        ) +
          0.5 * (vec_Z[j] * S_j).cwiseProduct(vec_Z[j]).rowwise().sum();

        total_par_looped += log_chol_par_per_block[k] + terms_per_block[k];

      }

    }

    total_ranef_blocks_looped += blocks_per_ranef[k];
    S_T.transposeInPlace();

    vec_S_by_ranef.push_back(S_T);

  }

  return vec_S_by_ranef;

}


Eigen::VectorXd get_grad_pois_glmm(
    Eigen::VectorXd& par,
    Eigen::VectorXd& par_scaling,
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

  std::vector<Eigen::MatrixXd> vec_S_by_ranef = get_link(
    par_scaled,
    vec_Z,
    y_nz_idx,
    blocks_per_ranef,
    log_chol_par_per_block,
    terms_per_block,
    link,
    S_by_block
  );

  int block_ctr = 0;
  int total_par_looped = 0;
  int total_ranef_blocks_looped = 0;
  int Sigma_start_idx = n_m_par + n_log_chol_par + n_b_par;
  int m_idx;

  Eigen::VectorXd iter_link;
  Eigen::MatrixXd Sigma;
  Eigen::MatrixXd Sigma_inv;

  // loop over each random effect block
  for (int k = 0; k < terms_per_block.size(); k++) {

    m_idx = 0;

    if (terms_per_block[k] == 1) {

      Sigma = get_S_1d(
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
          Zty(j),
          vec_Z[j],
          z2,
          par.segment(total_par_looped, 2), // parameters in the order (m1, ls1)
          par_scaling.segment(total_par_looped, 2),
          Sigma(0, 0),
          iter_link
        );

        m(m_idx) = par(total_par_looped);
        m_idx += 1;
        total_par_looped += 2;

      }

      Eigen::VectorXd s2 = vec_S_by_ranef[k].row(0);

      grad(Sigma_start_idx) = single_var_comp_1D_grad_glmm(
        m,
        s2,
        par_scaling(Sigma_start_idx),
        par.segment(Sigma_start_idx, 1)
      );

      Sigma_start_idx += 1;

    } else {

      int par_per_block = log_chol_par_per_block[k] + terms_per_block[k];

      Sigma = get_S(
        par_scaled.segment(Sigma_start_idx, log_chol_par_per_block[k]),
        log_chol_par_per_block[k]
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
          Zty.segment(total_par_looped, terms_per_block[k]),
          vec_Z[j],
          par.segment(total_par_looped, par_per_block), // parameters in the order (m1, ls1)
          par_scaling.segment(total_par_looped, par_per_block),
          Sigma_inv,
          iter_link,
          terms_per_block[k]
        );

        M_T.col(m_idx) = par.segment(total_par_looped, terms_per_block[k]);
        m_idx += 1;
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

  return grad;

}
