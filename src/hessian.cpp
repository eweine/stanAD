#include <RcppEigen.h>
#include "hessian.h"

Eigen::MatrixXd hess_inv_1D_pois_glmm_cpp(
    const double& sum_yz,
    const Eigen::VectorXd& z,
    double m,      // scalar mean parameter
    double log_s,  // log scalar standard deviation parameter
    double s2,
    const double& sig2,   // variance of prior
    Eigen::VectorXd exp_term // vector of offsets from other parameters
) {
  // Define z2 and z3
  Eigen::VectorXd z2 = z.array().square();    // elementwise z * z
  Eigen::VectorXd z3 = z2.cwiseProduct(z);

  double df2dm2 = -(exp_term.dot(z2) + (1 / sig2));
  double df2dlog_s2 = -((exp_term.array() * (z2.array() * std::exp(2 * log_s)).square()).sum() +
    2.0 * exp_term.dot(z2) * std::exp(2 * log_s) +
    2 * (std::exp(2 * log_s) / sig2));
  double df2dmdlog_s = -(exp_term.dot(z3) * std::exp(2 * log_s));

  Eigen::Matrix2d H_inv;
  double detH = df2dm2 * df2dlog_s2 - std::pow(df2dmdlog_s, 2);
  H_inv << df2dlog_s2, -df2dmdlog_s, -df2dmdlog_s, df2dm2;
  H_inv /= detH;

  return H_inv;
}
