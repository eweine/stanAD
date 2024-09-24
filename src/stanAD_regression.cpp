// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math/mix.hpp> // stuff from mix/ must come first
#include <stan/math.hpp>     // finally pull in everything from rev/ and prim/
#include <Rcpp.h>
#include <RcppEigen.h>
#include <type_traits>
#include "stanAD_regression.h"
#include "utils.h"


void single_newton_multiD_pois_glmm_cpp(
    const Eigen::VectorXd& Zty,
    const Eigen::MatrixXd& Z,
    Eigen::VectorXd& m,
    Eigen::VectorXd& S_log_chol,
    const std::vector<int> log_chol_diag_idx,
    Eigen::MatrixXd& S,
    const Eigen::MatrixXd& Sigma_inv,
    Eigen::VectorXd& link_offset
) {

  // This will be the code that will take advantage of the stan stuff
  Eigen::VectorXd ZSZT_diag = (Z * S).cwiseProduct(Z).rowwise().sum();
  Eigen::VectorXd link = Z * m + 0.5 * ZSZT_diag;

  // Update 'link_offset'
  link_offset = link_offset - link;
  Eigen::VectorXd exp_link_offset = link_offset.array().exp();

  int n_ranef_par = m.size();
  int n_log_chol_par = S_log_chol.size();
  Eigen::VectorXd par_vals(n_ranef_par + n_log_chol_par);
  par_vals << m, S_log_chol;

  double fx;
  Eigen::VectorXd grad;
  Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic> H;

  stan::math::hessian(
    [&Sigma_inv, &exp_link_offset, &Z, &n_ranef_par](auto par) {

      //return stan::math::sum(par.segment(0, n_ranef_par));
      using ScalarType = typename std::decay_t<decltype(par)>::Scalar;

      Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> Lfv(n_ranef_par, n_ranef_par);
      Lfv.setZero();
      //
      int index_s = n_ranef_par;
      // Fill the lower triangular matrix L with the log-Cholesky parameterization
      // in column major order
      for (int k = 0; k < n_ranef_par; ++k) {
        for (int l = k; l < n_ranef_par; ++l) {

          if (l == k) {
            Lfv(l, k) = stan::math::exp(par(index_s)); // exponentiate diagonal elements
          } else {
            Lfv(l, k) = par(index_s); // off-diagonal elements remain the same
          }
          index_s++;
        }
      }

      Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> Sfv = stan::math::tcrossprod(Lfv);
      return 0.5 * stan::math::trace(Sfv * Sigma_inv) +
        stan::math::dot_product(
          exp_link_offset,
          stan::math::exp(
            Z * par.segment(0, n_ranef_par) +
              0.5 * (Z * Sfv).cwiseProduct(Z).rowwise().sum()
          )
        );

    },
    par_vals, fx, grad, H);

  // Now, I want to make up for terms not in the stan code
  Eigen::VectorXd Sig_inv_m = Sigma_inv * m;
  fx += 0.5 * m.dot(Sig_inv_m) - Zty.dot(m) - par_vals(log_chol_diag_idx).sum();
  grad.segment(0, n_ranef_par) += Sig_inv_m - Zty;
  grad(log_chol_diag_idx).array() -= 1;
  H.topLeftCorner(n_ranef_par, n_ranef_par) += Sigma_inv;
  //Rprintf("Got gradient and hessian\n");
  //Rprintf("current_elbo = %f\n", fx);

  //
  // // I think this is about as efficient as I can get the code
  // // The next step is to carry out the line search
  Eigen::VectorXd dir = -H.inverse() * grad;
  Eigen::VectorXd m_proposed;
  Eigen::VectorXd S_log_chol_proposed;
  Eigen::VectorXd par_proposed;
  Eigen::MatrixXd L(n_ranef_par, n_ranef_par);
  Eigen::MatrixXd S_proposed(n_ranef_par, n_ranef_par);
  Eigen::VectorXd link_proposed;
  L.setZero();
  double elbo_proposed;
  int index;
  //
  double dec_const = dir.dot(grad);
  double cc = 1e-4;
  bool step_accepted = false;

  double alpha = 1;
  double beta = 0.25;
  //
  while (!step_accepted) {
    par_proposed = par_vals + alpha * dir;
    m_proposed = par_proposed.segment(0, n_ranef_par);
    S_log_chol_proposed = par_proposed.segment(n_ranef_par, n_log_chol_par);

    elbo_proposed = 0;
    //Rprintf("Getting L\n");
    index = 0;
    // Fill the lower triangular matrix L with the log-Cholesky parameterization
    // in column major order
    for (int j = 0; j < n_ranef_par; ++j) {
      for (int i = j; i < n_ranef_par; ++i) {

        if (i == j) {
          // Account for trace term
          elbo_proposed -= S_log_chol_proposed(index);
          L(i, j) = std::exp(S_log_chol_proposed(index)); // exponentiate diagonal elements
        } else {
          L(i, j) = S_log_chol_proposed(index); // off-diagonal elements remain the same
        }
        index++;
      }
    }

    //Rprintf("Finished getting L\n");

    S_proposed = L * L.transpose();
    //Rprintf("Printing S_proposed\n");
    //printMatrix(S_proposed);
    // Now, I should be able to calculate the proposed_elbo
    link_proposed = Z * m_proposed + 0.5 * (Z * S_proposed).cwiseProduct(Z).rowwise().sum();

    elbo_proposed += exp_link_offset.dot(link_proposed.array().exp().matrix()) -
      Zty.dot(m_proposed) + 0.5 * (
        m_proposed.dot(Sigma_inv * m_proposed) + (S_proposed * Sigma_inv).diagonal().sum()
      );

    //Rprintf("Proposed elbo = %f\n", elbo_proposed);

    if (elbo_proposed <= fx + cc * alpha * dec_const) {
      step_accepted = true;
      m = m_proposed;
      S_log_chol = S_log_chol_proposed;
      S = S_proposed;
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



// Rcpp::List single_newton_multiD_pois_glmm_cpp_testing(
//     const Eigen::VectorXd& Zty,
//     const Eigen::MatrixXd& Z,
//     Eigen::VectorXd& m,
//     Eigen::VectorXd& S_log_chol,
//     const std::vector<int> log_chol_diag_idx,
//     Eigen::MatrixXd& S,
//     const Eigen::MatrixXd& Sigma_inv,
//     Eigen::VectorXd& link_offset
// ) {
//
//   single_newton_multiD_pois_glmm_cpp(
//     Zty,
//     Z,
//     m,
//     S_log_chol,
//     log_chol_diag_idx,
//     S,
//     Sigma_inv,
//     link_offset
//   );
//
//   Rcpp::List out;
//   out["m"] = m;
//   out["S"] = S;
//   out["S_log_chol"] = S_log_chol;
//   return out;
//
// }
//
