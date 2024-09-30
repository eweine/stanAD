#ifndef LRVB_H
#define LRVB_H

#include <RcppEigen.h>

Eigen::MatrixXd get_lrvb_pois_glmm_mfvb(
    Eigen::VectorXd& m,
    Eigen::VectorXd& log_s,
    Eigen::VectorXd& b,
    Eigen::VectorXd& sigma2_inv,
    const std::vector<int>& blocks_per_ranef,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    const std::vector<int>& Z_i,
    const std::vector<int>& Z_j,
    const std::vector<double>& Z_x
);

#endif
