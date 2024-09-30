#ifndef HVP_H
#define HVP_H

#include <RcppEigen.h>

Eigen::VectorXd pois_glmm_mfvb_hvp(
    const Eigen::VectorXd& par_vals,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& v,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    const Eigen::SparseMatrix<double>& Z,
    const Eigen::SparseMatrix<double>& Z2,
    const std::vector<int>& blocks_per_ranef,
    int& n_ranef_par,
    int& n_fixef_par
);

#endif
