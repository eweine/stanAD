#include <RcppEigen.h>
#include "regression.h"
#include "utils.h"

// [[Rcpp::depends(RcppEigen)]]

void single_newton_1D_pois_glmm_cpp(
    const double& sum_yz,
    const Eigen::VectorXd& z,
    double& m,      // scalar mean parameter
    double& log_s,  // log scalar standard deviation parameter
    double& s2,
    const double& sig2,   // variance of prior
    Eigen::VectorXd& link_offset // vector of offsets from other parameters
) {
  // Define z2 and z3
  Eigen::VectorXd z2 = z.array().square();    // elementwise z * z
  Eigen::VectorXd z3 = z2.cwiseProduct(z);

  Eigen::VectorXd link = z * m + 0.5 * z2 * s2;
  Eigen::VectorXd link_exp = link.array().exp();

  link_offset = link_offset - link;
  Eigen::VectorXd exp_link_offset = link_offset.array().exp();
  Eigen::VectorXd exp_term = exp_link_offset.array() * link_exp.array();

  double current_lik = -sum_yz * m + exp_term.sum() +
    0.5 * (1 / sig2) * (m * m + s2) - log_s;

  //Rprintf("Current elbo = %f\n", current_lik);

  double dfdm = -sum_yz + exp_term.dot(z) + (m / sig2);
  double df2dm2 = exp_term.dot(z2) + (1 / sig2);
  double dfdlog_s = exp_term.dot(z2) * std::exp(2 * log_s) +
    (std::exp(2 * log_s) / sig2) - 1;
  double df2dlog_s2 = (exp_term.array() * (z2.array() * std::exp(2 * log_s)).square()).sum() +
    2.0 * exp_term.dot(z2) * std::exp(2 * log_s) +
    2 * (std::exp(2 * log_s) / sig2);
  double df2dmdlog_s = exp_term.dot(z3) * std::exp(2 * log_s);

  // Compute Newton step
  Eigen::VectorXd g(2);
  g << dfdm, dfdlog_s;

  Eigen::MatrixXd H_inv(2, 2);
  double detH = df2dm2 * df2dlog_s2 - std::pow(df2dmdlog_s, 2);
  H_inv << df2dlog_s2, -df2dmdlog_s, -df2dmdlog_s, df2dm2;
  H_inv /= detH;

  Eigen::VectorXd dir = -H_inv * g;
  Eigen::Matrix<double, 2, 1> par(2);
  par << m, log_s;

  double dec_const = dir.dot(g);
  double cc = 1e-4;
  bool step_accepted = false;
  double m_proposed;
  double log_s_proposed;
  double s2_proposed;
  double lik_proposed;

  double alpha = 1;
  double beta = 0.25;

  while (!step_accepted) {
    Eigen::VectorXd par_proposed = par + alpha * dir;
    m_proposed = par_proposed(0);
    log_s_proposed = par_proposed(1);
    s2_proposed = std::pow(std::exp(log_s_proposed), 2);
    Eigen::VectorXd link_proposed = z * m_proposed + 0.5 * z2 * s2_proposed;
    Eigen::VectorXd exp_term_proposed = exp_link_offset.array() * link_proposed.array().exp();
    lik_proposed = -sum_yz * m_proposed + exp_term_proposed.sum() +
      0.5 * (1 / sig2) * (m_proposed * m_proposed + s2_proposed) - log_s_proposed;

    //Rprintf("Proposed elbo = %f\n", lik_proposed);

    if (lik_proposed <= current_lik + cc * alpha * dec_const) {
      step_accepted = true;
      m = m_proposed;
      log_s = log_s_proposed;
      s2 = s2_proposed;
      par = par_proposed;
      link_offset = link_offset + link_proposed;
    } else {
      alpha *= beta;
      if (alpha < 1e-12) {
        link_offset = link_offset + link;
        return;
      }
    }
  }

  return;
}



// Rcpp::List single_newton_1D_pois_glmm_cpp_testing(
//    const double& sum_yz,
//    const Eigen::VectorXd& z,
//    double& m,      // scalar mean parameter
//    double& log_s,  // log scalar standard deviation parameter
//    double& s2,
//    const double& sig2,   // variance of prior
//    Eigen::VectorXd& link_offset // vector of offsets from other parameters
// ) {
//
//   single_newton_1D_pois_glmm_cpp(
//     sum_yz,
//     z,
//     m,      // scalar mean parameter
//     log_s,  // log scalar standard deviation parameter
//     s2,
//     sig2,   // variance of prior
//     link_offset // vector of offsets from other parameters
//   );
//
//   Rcpp::List out;
//   out["m"] = m;
//   out["log_s"] = log_s;
//   return out;
//
// }
//

void single_newton_mod_pois_reg(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& Xty,
    Eigen::VectorXd& link_offset,
    Eigen::VectorXd& b
) {

  Eigen::VectorXd eta = X * b;
  link_offset -= eta;
  Eigen::VectorXd a = link_offset.array().exp();
  Eigen::VectorXd exp_eta = eta.array().exp();
  double current_lik = -Xty.dot(b) + a.dot(exp_eta);
  //Rprintf("Current fixef lik = %f\n", current_lik);

  //Rprintf("Printing y_tilde\n");
  Eigen::VectorXd y_tilde = a.array() * exp_eta.array();
  //printVector(y_tilde);
  //Rprintf("Printing g\n");
  Eigen::VectorXd g = (X.transpose() * y_tilde) - Xty;
  Rprintf("Size of g = %li\n", g.size());
  //printVector(g);
  //Rprintf("Printing H\n");
  Eigen::MatrixXd H = X.transpose() * y_tilde.asDiagonal() * X;
  //printMatrix(H);
  Eigen::VectorXd dir = -H.inverse() * g;
  Rprintf("Size of dir = %li\n", dir.size());
  Eigen::VectorXd b_proposed;

  double dec_const = dir.dot(g);
  double cc = 1e-4;

  double alpha = 1;
  double beta = 0.25;

  Eigen::VectorXd eta_proposed;
  double lik_proposed;

  while (alpha >= 1e-12) {

    b_proposed = b + alpha * dir;
    eta_proposed = X * b_proposed;
    lik_proposed = -Xty.dot(b_proposed) +
      a.dot(eta_proposed.array().exp().matrix());

    if (lik_proposed <= current_lik + cc * alpha * dec_const) {

      link_offset += eta_proposed;
      b = b_proposed;
      return;

    } else {

      alpha *= beta;

    }

  }

  // if no step is accepted, just return previous iterate
  link_offset += eta;
  return;

}
