#ifndef HESSIAN_H
#define HESSIAN_H

#include <RcppEigen.h>

Eigen::MatrixXd hess_inv_1D_pois_glmm_cpp(
    const double& sum_yz,
    const Eigen::VectorXd& z,
    double m,      // scalar mean parameter
    double log_s,  // log scalar standard deviation parameter
    double s2,
    const double& sig2,   // variance of prior
    Eigen::VectorXd exp_term // vector of offsets from other parameters
);

#endif
