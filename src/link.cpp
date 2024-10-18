#include <RcppEigen.h>
#include "utils.h"

std::vector<Eigen::MatrixXd> get_link_pois_glmm(
    Eigen::VectorXd& scaled_par,
    Eigen::MatrixXd& X,
    std::vector<Eigen::MatrixXd>& vec_Z,
    std::vector<std::vector<int>>& y_nz_idx,
    const std::vector<int>& blocks_per_ranef,
    const std::vector<int>& log_chol_par_per_block,
    const std::vector<int>& terms_per_block,
    Eigen::VectorXd& link,
    std::vector<Eigen::MatrixXd>& S_by_block,
    int& fixef_start,
    int& n_b
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

        S_j = get_sigma2_from_log_sigma(scaled_par(total_par_looped + 1));
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

        S_j = get_Sigma_from_log_chol(
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

  link += X * scaled_par.segment(fixef_start, n_b);

  return vec_S_by_ranef;

}
