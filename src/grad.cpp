// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math/mix.hpp> // stuff from mix/ must come first
#include <stan/math.hpp>     // finally pull in everything from rev/ and prim/
#include <Rcpp.h>
#include <RcppEigen.h>
#include <type_traits>
#include "grad.h"


Eigen::VectorXd single_local_block_multiD_grad_pois_glmm(
    const Eigen::VectorXd Zty,
    const Eigen::MatrixXd& Z,
    Eigen::VectorXd par_vals, // parameters in the order (m1, m2, ls1, ls2, ls3)
    Eigen::VectorXd par_scaling,
    Eigen::MatrixXd& Sigma_inv,
    Eigen::VectorXd link,
    int n_m_par
) {

  double fx;
  Eigen::VectorXd grad_fx;
  Eigen::VectorXd exp_link_prod = link.array().exp();

  stan::math::gradient(
    [&Sigma_inv, &exp_link_prod, &Zty, &Z, &par_scaling, &n_m_par](auto par) {

      using ScalarType = typename std::decay_t<decltype(par)>::Scalar;

      Eigen::Vector<ScalarType, Eigen::Dynamic> par_scaled = par.cwiseProduct(
        par_scaling
      );

      ScalarType elbo = 0.0;

      Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> Lfv(n_m_par, n_m_par);
      Lfv.setZero();
      //
      int index_s = n_m_par;
      // // Fill the lower triangular matrix L with the log-Cholesky parameterization
      // // in column major order
      for (int k = 0; k < n_m_par; ++k) {
        for (int l = k; l < n_m_par; ++l) {

          if (l == k) {
            elbo -= par_scaled(index_s);
            Lfv(l, k) = stan::math::exp(par_scaled(index_s)); // exponentiate diagonal elements
          } else {
            Lfv(l, k) = par_scaled(index_s); // off-diagonal elements remain the same
          }
          index_s++;
        }
      }

      Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> Sfv = Lfv * Lfv.transpose();
      //Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> S_Sig = Sfv * Sigma_inv;
      elbo += -stan::math::dot_product(Zty, par_scaled.segment(0, n_m_par)) +
        0.5 * Sigma_inv.cwiseProduct(Sfv).sum() +
        stan::math::dot_product(
          exp_link_prod,
          stan::math::exp(
            Z * par_scaled.segment(0, n_m_par) +
              0.5 * (Z * Sfv).cwiseProduct(Z).rowwise().sum()
          )
        ) +
        0.5 * stan::math::dot_product(
          par_scaled.segment(0, n_m_par),
          Sigma_inv * par_scaled.segment(0, n_m_par)
        );

      return elbo;

    },
    par_vals, fx, grad_fx);

  return grad_fx;

}

Eigen::VectorXd fixef_grad_pois_glmm(
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    Eigen::VectorXd b,
    Eigen::VectorXd b_scaling,
    Eigen::VectorXd link
) {

  double fx;
  Eigen::VectorXd grad_fx;
  Eigen::VectorXd exp_link_prod = link.array().exp();

  stan::math::gradient(
    [&Xty, &exp_link_prod, &X, &b_scaling](auto b_v) {

      using ScalarType = typename std::decay_t<decltype(b_v)>::Scalar;

      Eigen::Vector<ScalarType, Eigen::Dynamic> b_scaled = b_v.cwiseProduct(
        b_scaling
      );

      return -stan::math::dot_product(Xty, b_scaled) +
        stan::math::dot_product(
          exp_link_prod,
          stan::math::exp(
            X * b_scaled
          )
        );

    },
    b, fx, grad_fx);

  return grad_fx;

}


Eigen::VectorXd single_local_block_1D_grad_pois_glmm(
    const double Zty,
    const Eigen::VectorXd& z,
    const Eigen::VectorXd& z2,
    Eigen::Vector<double, 2> par_vals, // parameters in the order (m1, ls1)
    Eigen::Vector<double, 2> par_scaling,
    double sigma2,
    Eigen::VectorXd& link
) {

  double fx;
  // could fix this to 2d
  Eigen::VectorXd grad_fx;
  Eigen::VectorXd exp_link_prod = link.array().exp();

  stan::math::gradient(
    [&sigma2, &exp_link_prod, &Zty, &z, &z2, &par_scaling](auto par) {

      using ScalarType = typename std::decay_t<decltype(par)>::Scalar;
      Eigen::Vector<ScalarType, Eigen::Dynamic> par_scaled = par.cwiseProduct(
        par_scaling
        );

      ScalarType s2 = stan::math::square(stan::math::exp(par_scaled(1)));
      Eigen::Vector<ScalarType, Eigen::Dynamic> exp_arg = z * par_scaled(0) + 0.5 * z2 * s2;

      return -Zty * par_scaled(0) +
          exp_arg.array().exp().matrix().dot(exp_link_prod) +
        0.5 * (1 / sigma2) * (stan::math::square(par_scaled(0)) + s2) -
        par_scaled(1);

    },
    par_vals, fx, grad_fx);

  return grad_fx;

}


double single_var_comp_1D_grad_glmm(
    Eigen::VectorXd& m,
    Eigen::VectorXd& s2,
    double par_scaling,
    Eigen::VectorXd log_sigma
) {

  double fx;
  Eigen::VectorXd grad_fx;
  // These must be the properly scaled values
  double par_sum = 0.5 * (
    m.array().square().sum() + s2.sum()
    );

  double det_scaling = static_cast<double>(m.size());

  stan::math::gradient(
    [&par_sum, &par_scaling, &det_scaling](auto par) {

      using ScalarType = typename std::decay_t<decltype(par)>::Scalar;
      ScalarType log_sigma_scaled = par(0) * par_scaling;
      ScalarType sigma2_inv = 1 / stan::math::square(stan::math::exp(log_sigma_scaled));


      return det_scaling * log_sigma_scaled + sigma2_inv * par_sum;

    },
    log_sigma, fx, grad_fx);

  return grad_fx(0);

}


Eigen::VectorXd single_var_comp_multiD_grad_glmm(
    Eigen::MatrixXd& M,
    Eigen::MatrixXd& S,
    Eigen::VectorXd par_scaling,
    Eigen::VectorXd Sigma_log_chol,
    int Sigma_d
) {

  double fx;
  Eigen::VectorXd grad_fx;

  double det_scaling = static_cast<double>(M.rows());

  stan::math::gradient(
    [&M, &S, &par_scaling, &Sigma_d, &det_scaling](auto par) {

      using ScalarType = typename std::decay_t<decltype(par)>::Scalar;
      Eigen::Vector<ScalarType, Eigen::Dynamic> par_scaled = par.cwiseProduct(
        par_scaling
      );

      ScalarType elbo = 0.0;

      Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> Lfv(Sigma_d, Sigma_d);
      Lfv.setZero();

      int index_Sigma = 0;
      // Fill the lower triangular matrix L with the log-Cholesky parameterization
      // in column major order
      for (int k = 0; k < Sigma_d; ++k) {
        for (int l = k; l < Sigma_d; ++l) {

          if (l == k) {
            elbo += det_scaling * par_scaled(index_Sigma);
            Lfv(l, k) = stan::math::exp(par_scaled(index_Sigma)); // exponentiate diagonal elements
          } else {
            Lfv(l, k) = par_scaled(index_Sigma); // off-diagonal elements remain the same
          }
          index_Sigma++;
        }
      }

      Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> Sigma_inv = stan::math::inverse(
        stan::math::tcrossprod(Lfv)
      );

      elbo += 0.5 * (
        stan::math::sum(M.cwiseProduct(M * Sigma_inv)) +
          stan::math::sum(S * Sigma_inv.reshaped())
      );

      return elbo;

    },
    Sigma_log_chol, fx, grad_fx);

  return grad_fx;

}
