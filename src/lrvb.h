#ifndef LRVB_H
#define LRVB_H

#include <RcppEigen.h>

Eigen::MatrixXd get_lrvb_pois_glmm_mfvb(
    Eigen::VectorXd& par_vals,
    const std::vector<int>& blocks_per_ranef,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    Eigen::SparseMatrix<double>& Z,
    Eigen::SparseMatrix<double>& Z2,
    int n_ranef_par,
    int n_fixef_par
);

#endif
