// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math/mix.hpp> // stuff from mix/ must come first
#include <stan/math.hpp>     // finally pull in everything from rev/ and prim/
#include <Rcpp.h>
#include <RcppEigen.h>       // do this AFTER including stuff from stan/math

// [[Rcpp::plugins(cpp17)]]


// [[Rcpp::export]]
Eigen::VectorXd get_elbo_hvp(
    const Eigen::VectorXd& par_vals,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& v,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    const Eigen::SparseMatrix<double>& Z,
    const Eigen::SparseMatrix<double>& Z2,
    const std::vector<int>& blocks_per_ranef,
    const std::vector<int>& terms_per_block,
    int& n_ranef_par,
    int& n_fixef_par
) {

  double fx;
  Eigen::Matrix< double, Eigen::Dynamic, 1 > Hv;
  stan::math::hessian_times_vector(
    [&Zty, &Xty, &Z, &Z2, &blocks_per_ranef, &terms_per_block, &n_ranef_par, &n_fixef_par, &X](auto par) {

      stan::math::fvar<stan::math::var> elbo = stan::math::dot_product(Zty, par.segment(0, n_ranef_par)) +
        stan::math::dot_product(Xty, par.segment(2*n_ranef_par, n_fixef_par)) -
        stan::math::sum(
          stan::math::exp(
            Z * par.segment(0, n_ranef_par) + X * par.segment(2*n_ranef_par, n_fixef_par) +
              0.5 * Z2 * stan::math::square(stan::math::exp(par.segment(n_ranef_par, n_ranef_par)))
          )
        ) + stan::math::sum(par.segment(n_ranef_par, n_ranef_par));

      // great, so the above seems to work
      // the next step will be to add in the other components

      int cols_iterated_through = 0;
      int log_chol_par_idx = 2 * n_ranef_par + n_fixef_par;

      // loop over each random effect block
      for (int k = 0; k < terms_per_block.size(); k++) {

        Eigen::Matrix<stan::math::fvar<stan::math::var>, Eigen::Dynamic, Eigen::Dynamic> L(terms_per_block[k], terms_per_block[k]);
        L.setZero();

        // Fill the lower triangular matrix L with the log-Cholesky parameterization
        // in column major order
        for (int j = 0; j < terms_per_block[k]; ++j) {
          for (int i = j; i < terms_per_block[k]; ++i) {

            if (i == j) {
              L(i, j) = stan::math::exp(par(log_chol_par_idx)); // exponentiate diagonal elements
            } else {
              L(i, j) = par(log_chol_par_idx); // off-diagonal elements remain the same
            }
            log_chol_par_idx++;
          }
        }

        Eigen::Matrix<stan::math::fvar<stan::math::var>, Eigen::Dynamic, Eigen::Dynamic> Sigma = L * L.transpose();
        elbo -= 0.5 * static_cast<double>(blocks_per_ranef[k]) *
          stan::math::log_determinant(Sigma);

        Eigen::Matrix<stan::math::fvar<stan::math::var>, Eigen::Dynamic, Eigen::Dynamic> Sigma_inv = stan::math::inverse(Sigma);

        elbo -= 0.5 * (stan::math::sum(
          stan::math::square(
            stan::math::exp(
              par.segment(
                n_ranef_par + cols_iterated_through, blocks_per_ranef[k] * terms_per_block[k]
              )
            )
          ).reshaped(terms_per_block[k], blocks_per_ranef[k]).rowwise().sum().array() * Sigma_inv.diagonal().array()));

        elbo -= 0.5 * stan::math::sum(
          (
              par.segment(
                cols_iterated_through, blocks_per_ranef[k] * terms_per_block[k]
              ).reshaped(terms_per_block[k], blocks_per_ranef[k]).transpose() * Sigma_inv
          ).array() * par.segment(
              cols_iterated_through, blocks_per_ranef[k] * terms_per_block[k]
          ).reshaped(terms_per_block[k], blocks_per_ranef[k]).transpose().array()
        );

        //

        cols_iterated_through += blocks_per_ranef[k] * terms_per_block[k];

      }

      return elbo;

    },
    par_vals, v, fx, Hv);

  return Hv;

}
