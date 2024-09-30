#ifndef UTILS_H
#define UTILS_H

#include <RcppEigen.h>

std::vector<int> get_n_nz_terms(
    const std::vector<int> n_nz_terms_per_col,
    const std::vector<int> blocks_per_ranef,
    const std::vector<int> terms_per_block
);

std::vector<Eigen::MatrixXd> create_ranef_Z(
    const std::vector<int>& n_nz_terms,    // number of nonzero terms for each ranef coefficient
    const std::vector<int>& blocks_per_ranef,  // number of coefficients per ranef (e.g. 50)
    const std::vector<int>& terms_per_block,  // number of terms in each random effect
    const std::vector<double>& values        // values of Z
);

std::vector<Eigen::MatrixXd> initialize_Sigma(
    const std::vector<int>& blocks_per_ranef
);

std::vector<std::vector<int>> create_y_nz_idx(
    const std::vector<int>& n_nz_terms,    // number of nonzero terms for each ranef coefficient
    const std::vector<int>& blocks_per_ranef,  // number of coefficients per ranef (e.g. 50)
    const std::vector<int>& terms_per_block,  // number of terms in each random effect
    const std::vector<int>& Z_i              // values of Z
);

std::vector<Eigen::MatrixXd> initialize_S(
    const std::vector<int>& blocks_per_ranef,  // number of coefficients per ranef (e.g. 50)
    const std::vector<int>& terms_per_block
);

std::vector<std::vector<int>> get_log_chol_diag_idx_per_ranef(
    const std::vector<int>& terms_per_block
);

void printVector(const Eigen::VectorXd& vector);

void printMatrix(const Eigen::MatrixXd& matrix);

void create_Z_and_Z2(const std::vector<int>& Z_i,
                     const std::vector<int>& Z_j,
                     const std::vector<double>& Z_x,
                     Eigen::SparseMatrix<double>& Z,
                     Eigen::SparseMatrix<double>& Z2,
                     int rows, int cols);

#endif
