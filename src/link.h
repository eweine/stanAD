#ifndef LINK_H
#define LINK_H

#include <RcppEigen.h>

std::vector<Eigen::MatrixXd> get_link_pois_glmm(
    Eigen::VectorXd& scaled_par,
    Eigen::MatrixXd& X,
    std::vector<Eigen::MatrixXd>& vec_Z,
    std::vector<std::vector<int>>& y_nz_idx,
    const std::vector<int>& blocks_per_ranef,
    const std::vector<int>& log_chol_par_per_block,
    const std::vector<int>& terms_per_block,
    Eigen::VectorXd& link,
    std::vector<Eigen::MatrixXd>& S_by_block,
    int& fixef_start,
    int& n_b
);

#endif
