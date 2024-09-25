#include "stanAD_regression.h"
#include <RcppEigen.h>
#include "regression.h"

void fpiter_pois_glmm(
    Eigen::VectorXd& m,
    Eigen::VectorXd& S_log_chol,
    std::vector<Eigen::MatrixXd>& S, // maybe change to a vector of matrices?
    Eigen::VectorXd& b,
    Eigen::VectorXd& link_offset,
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& Zty,
    const Eigen::VectorXd& Xty,
    const std::vector<int>& blocks_per_ranef,
    const std::vector<int>& log_chol_par_per_block,
    const std::vector<int>& terms_per_block,
    const std::vector<std::vector<int>>& log_chol_diag_idx_per_ranef,
    std::vector<Eigen::MatrixXd>& vec_Z,
    std::vector<std::vector<int>>& y_nz_idx,
    std::vector<Eigen::MatrixXd>& Sigma
) {

  // calling the above functions, I can take a single step
  int total_ranef_blocks_looped = 0;
  int m_par_iterated_through = 0;
  int log_chol_par_iterated_through = 0;
  Eigen::VectorXd iter_link_offset;

  // loop over each random effect block
  for (int k = 0; k < terms_per_block.size(); k++) {

    if (terms_per_block[k] == 1) {

      //Rprintf("Entering code for single term per block\n");
      double sig2 = (Sigma[k])(0, 0);
      //Rprintf("sig2 = %f\n", sig2);
      Eigen::VectorXd new_par;
      double iter_m;
      double iter_log_s;
      double iter_s2;
      double sig2_new = 0;

      for (int j = total_ranef_blocks_looped; j < total_ranef_blocks_looped + blocks_per_ranef[k]; j++) {

        //Rprintf("j = %i\n", j);

        iter_link_offset = link_offset(y_nz_idx[j]);
        iter_m = m(m_par_iterated_through);
        iter_log_s = S_log_chol(log_chol_par_iterated_through);
        iter_s2 = (S[j])(0, 0);

        // I may be able to update in place with () references
        // I should check this later
        single_newton_1D_pois_glmm_cpp(
           Zty(m_par_iterated_through),
           vec_Z[j],
           iter_m,
           iter_log_s,
           iter_s2,
           sig2,
           iter_link_offset
        );

        m(m_par_iterated_through) = iter_m;
        S_log_chol(log_chol_par_iterated_through) = iter_log_s;
        (S[j])(0, 0) = iter_s2;
        link_offset(y_nz_idx[j]) = iter_link_offset;

        sig2_new += (iter_m * iter_m) + iter_s2;
        m_par_iterated_through += 1;
        log_chol_par_iterated_through += 1;

      }

      //Rprintf("Updating Sigma...\n");
      (Sigma[k])(0, 0) = sig2_new / (static_cast<double>(blocks_per_ranef[k]));
      //Rprintf("Done updating Sigma...\n");

    } else {

      Eigen::MatrixXd Sigma_inv = Sigma[k].inverse();
      Eigen::MatrixXd Sigma_new(terms_per_block[k], terms_per_block[k]);
      Sigma_new.setZero();
      Eigen::VectorXd iter_m;
      Eigen::VectorXd iter_log_chol;
      Eigen::MatrixXd iter_S;

      for (int j = total_ranef_blocks_looped; j < total_ranef_blocks_looped + blocks_per_ranef[k]; j++) {

        iter_link_offset = link_offset(y_nz_idx[j]);
        iter_m = m.segment(m_par_iterated_through, terms_per_block[k]);
        iter_log_chol = S_log_chol.segment(log_chol_par_iterated_through, log_chol_par_per_block[k]);
        iter_S = S[j];

        single_newton_multiD_pois_glmm_cpp(
          Zty.segment(m_par_iterated_through, terms_per_block[k]),
          vec_Z[j],
          iter_m,
          iter_log_chol,
          log_chol_diag_idx_per_ranef[k],
          iter_S,
          Sigma_inv,
          iter_link_offset
        );

        link_offset(y_nz_idx[j]) = iter_link_offset;
        m.segment(m_par_iterated_through, terms_per_block[k]) = iter_m;
        S_log_chol.segment(log_chol_par_iterated_through, log_chol_par_per_block[k]) = iter_log_chol;
        S[j] = iter_S;

        Sigma_new += iter_S + (iter_m * iter_m.transpose());
        m_par_iterated_through += terms_per_block[k];
        log_chol_par_iterated_through += log_chol_par_per_block[k];

      }

      Sigma[k] = Sigma_new / (static_cast<double>(blocks_per_ranef[k]));

    }

    total_ranef_blocks_looped += blocks_per_ranef[k];

  }


  //Rprintf("Updating fixed effects\n");
  //Rprintf("Size of b = %li\n", b.size());
  //Rprintf("Size of Xty = %li\n", Xty.size());
  // now, update fixed effects parameters
  single_newton_mod_pois_reg(
    X,
    Xty,
    link_offset,
    b
  );

  //Rprintf("Done updating fixed effects\n");
  //Rprintf("Size of b = %li\n", b.size());

  return;

}
