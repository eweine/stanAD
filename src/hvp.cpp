// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math/mix.hpp> // stuff from mix/ must come first
#include <stan/math.hpp>     // finally pull in everything from rev/ and prim/
#include <Rcpp.h>
#include <RcppEigen.h>
#include <type_traits>
#include "hvp.h"

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
) {

  // the organization of the parameters in the vector is as follows:
  // (m1, log_s1, m2, log_s2, ..., b1, ..., bp, sigma1^2, ..., sigmak^2)

  double fx;
  Eigen::Matrix< double, Eigen::Dynamic, 1 > Hv;
  stan::math::hessian_times_vector(
    [&Zty, &Xty, &Z, &Z2, &blocks_per_ranef, &n_ranef_par, &n_fixef_par, &X](auto par) {

      using ScalarType = typename std::decay_t<decltype(par)>::Scalar;

      Eigen::Matrix<ScalarType, Eigen::Dynamic, 2> ranef_mat = par.segment(
        0, 2 * n_ranef_par
      ).reshaped(2, n_ranef_par).transpose();

      Eigen::Vector<ScalarType, Eigen::Dynamic> s2 = stan::math::square(
        stan::math::exp(
          ranef_mat.col(1)
        )
      );

      ScalarType elbo = stan::math::dot_product(Zty, ranef_mat.col(0)) +
        stan::math::dot_product(Xty, par.segment(2*n_ranef_par, n_fixef_par)) -
        stan::math::sum(
          stan::math::exp(
            Z * ranef_mat.col(0) + X * par.segment(2*n_ranef_par, n_fixef_par) +
              0.5 * Z2 * s2
          )
        ) + stan::math::sum(ranef_mat.col(1));

      int cols_iterated_through = 0;

      // loop over each random effect block
      for (int k = 0; k < blocks_per_ranef.size(); k++) {

        elbo += -0.5 * ((1 / par(2*n_ranef_par + n_fixef_par + k)) * (
          stan::math::dot_self(
            ranef_mat.col(0).segment(cols_iterated_through, blocks_per_ranef[k])
          ) + stan::math::sum(
              s2.segment(cols_iterated_through, blocks_per_ranef[k])
          )
        ) + (static_cast<double>(blocks_per_ranef[k]) *
          stan::math::log(par(2*n_ranef_par + n_fixef_par + k)))
        );

        cols_iterated_through += blocks_per_ranef[k];

      }

      return -elbo;

    },
    par_vals, v, fx, Hv);

  return Hv;

}


Rcpp::List pois_glmm_mfvb_h_test(
    const Eigen::VectorXd& par_vals,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    const Eigen::SparseMatrix<double>& Z,
    const Eigen::SparseMatrix<double>& Z2,
    const std::vector<int>& blocks_per_ranef,
    int& n_ranef_par,
    int& n_fixef_par
) {

  // the organization of the parameters in the vector is as follows:
  // (m1, log_s1, m2, log_s2, ..., b1, ..., bp, sigma1^2, ..., sigmak^2)

  double fx;
  Eigen::VectorXd grad;
  Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic> H;
  stan::math::hessian(
    [&Zty, &Xty, &Z, &Z2, &blocks_per_ranef, &n_ranef_par, &n_fixef_par, &X](auto par) {

      using ScalarType = typename std::decay_t<decltype(par)>::Scalar;

      Eigen::Matrix<ScalarType, Eigen::Dynamic, 2> ranef_mat = par.segment(
        0, 2 * n_ranef_par
      ).reshaped(2, n_ranef_par).transpose();

      Eigen::Vector<ScalarType, Eigen::Dynamic> s2 = stan::math::square(
        stan::math::exp(
          ranef_mat.col(1)
        )
      );

      ScalarType elbo = stan::math::dot_product(Zty, ranef_mat.col(0)) +
        stan::math::dot_product(Xty, par.segment(2*n_ranef_par, n_fixef_par)) -
        stan::math::sum(
          stan::math::exp(
            Z * ranef_mat.col(0) + X * par.segment(2*n_ranef_par, n_fixef_par) +
              0.5 * Z2 * s2
          )
        ) + stan::math::sum(ranef_mat.col(1));

      int cols_iterated_through = 0;

      // loop over each random effect block
      for (int k = 0; k < blocks_per_ranef.size(); k++) {

        elbo += -0.5 * ((1 / par(2*n_ranef_par + n_fixef_par + k)) * (
          stan::math::dot_self(
            ranef_mat.col(0).segment(cols_iterated_through, blocks_per_ranef[k])
          ) + stan::math::sum(
              s2.segment(cols_iterated_through, blocks_per_ranef[k])
          )
        ) + (static_cast<double>(blocks_per_ranef[k]) *
          stan::math::log(par(2*n_ranef_par + n_fixef_par + k)))
        );

        cols_iterated_through += blocks_per_ranef[k];

      }

      return -elbo;

    },
    par_vals, fx, grad, H);

  Rcpp::List out;
  out["H"] = H;
  out["elbo"] = fx;
  out["grad"] = grad;

  return out;

}

