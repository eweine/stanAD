#ifndef REGRESSION_H
#define REGRESSION_H

#include <RcppEigen.h>

void single_newton_1D_pois_glmm_cpp(
    const double& sum_yz,
    const Eigen::VectorXd& z,
    double& m,      // scalar mean parameter
    double& log_s,  // log scalar standard deviation parameter
    double& s2,
    const double& sig2,   // variance of prior
    Eigen::VectorXd& link_offset // vector of offsets from other parameters
);

void single_newton_mod_pois_reg(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& Xty,
    Eigen::VectorXd& link_offset,
    Eigen::VectorXd& b
);

#endif
