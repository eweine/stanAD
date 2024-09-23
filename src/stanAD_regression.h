#ifndef STANAD_REGRESSION_H
#define STANAD_REGRESSION_H

#include <RcppEigen.h>

void single_newton_multiD_pois_glmm_cpp(
    const Eigen::VectorXd& Zty,
    const Eigen::MatrixXd& Z,
    Eigen::VectorXd& m,
    Eigen::VectorXd& S_log_chol,
    std::vector<int> log_chol_diag_idx,
    Eigen::MatrixXd& S,
    const Eigen::MatrixXd& Sigma_inv,
    Eigen::VectorXd& link_offset
);

#endif
