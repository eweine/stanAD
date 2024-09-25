#ifndef ELBO_H
#define ELBO_H

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
    std::vector<Eigen::MatrixXd>& Sigma
);

#endif
