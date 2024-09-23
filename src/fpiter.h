#ifndef FPITER_H
#define FPITER_H

#include <RcppEigen.h>

void fpiter_pois_glmm(
    Eigen::VectorXd& m,
    Eigen::VectorXd& S_log_chol,
    std::vector<Eigen::MatrixXd>& S, // maybe change to a vector of matrices?
    Eigen::VectorXd& b,
    Eigen::VectorXd& link_offset,
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const std::vector<int>& blocks_per_ranef,
    const std::vector<int>& log_chol_par_per_block,
    const std::vector<int>& terms_per_block,
    const std::vector<std::vector<int>>& log_chol_diag_idx_per_ranef,
    std::vector<Eigen::MatrixXd>& vec_Z,
    std::vector<std::vector<int>>& y_nz_idx,
    std::vector<Eigen::MatrixXd>& Sigma
);

#endif
