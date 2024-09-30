#include <RcppEigen.h>
#include "utils.h"
#include <numeric>

// [[Rcpp::depends(RcppEigen)]]

std::vector<int> get_n_nz_terms(
    const std::vector<int> n_nz_terms_per_col,
    const std::vector<int> blocks_per_ranef,
    const std::vector<int> terms_per_block
) {

  int tot = std::accumulate(blocks_per_ranef.begin(), blocks_per_ranef.end(), 0);
  std::vector<int> n_nz_terms(tot);

  int total_ranef_coefs_looped = 0;
  int j_start = 0;

  for (int ranef_idx = 0; ranef_idx < terms_per_block.size(); ranef_idx++) {

    // loop over all the coefficients in the random effect block
    for (int j = j_start; j < j_start + blocks_per_ranef[ranef_idx]; j++) {

      n_nz_terms[j] = n_nz_terms_per_col[total_ranef_coefs_looped];
      total_ranef_coefs_looped += terms_per_block[ranef_idx];

    }

    j_start += blocks_per_ranef[ranef_idx];

  }

  return n_nz_terms;

}


std::vector<Eigen::MatrixXd> create_ranef_Z(
    const std::vector<int>& n_nz_terms,    // number of nonzero terms for each ranef coefficient
    const std::vector<int>& blocks_per_ranef,  // number of coefficients per ranef (e.g. 50)
    const std::vector<int>& terms_per_block,  // number of terms in each random effect
    const std::vector<double>& values        // values of Z
) {

  // Preallocate memory for the vector
  std::vector<Eigen::MatrixXd> vectorOfMats;
  int tot = std::accumulate(blocks_per_ranef.begin(), blocks_per_ranef.end(), 0);
  vectorOfMats.reserve(tot);

  int values_ctr = 0;
  int total_ranef_coefs_looped = 0;

  // loop over each random effect block
  for (int ranef_idx = 0; ranef_idx < terms_per_block.size(); ranef_idx++) {

    // loop over all the coefficients in the random effect block
    for (int j = 0; j < blocks_per_ranef[ranef_idx]; j++) {

      // Create a matrix: n_nz_terms x terms_per_block
      Eigen::MatrixXd zmat(n_nz_terms[total_ranef_coefs_looped], terms_per_block[ranef_idx]);

      for (int k = 0; k < zmat.cols(); k++) {
        for (int i = 0; i < zmat.rows(); i++) {
          zmat(i, k) = values[values_ctr];
          values_ctr += 1;
        }
      }

      vectorOfMats.push_back(zmat);
      total_ranef_coefs_looped += 1;
    }
  }

  return vectorOfMats;
}

std::vector<Eigen::MatrixXd> initialize_Sigma(
    const std::vector<int>& terms_per_block
) {

  std::vector<Eigen::MatrixXd> vectorOfMats;
  vectorOfMats.reserve(terms_per_block.size());

  for (int i = 0; i < terms_per_block.size(); i++) {

    // Initialize with relatively large values on the diagonal
    vectorOfMats.push_back(
      3 * Eigen::MatrixXd::Identity(terms_per_block[i], terms_per_block[i])
    );

  }

  return vectorOfMats;

}

std::vector<std::vector<int>> create_y_nz_idx(
    const std::vector<int>& n_nz_terms,    // number of nonzero terms for each ranef coefficient
    const std::vector<int>& blocks_per_ranef,  // number of coefficients per ranef (e.g. 50)
    const std::vector<int>& terms_per_block,  // number of terms in each random effect
    const std::vector<int>& Z_i              // values of Z
) {

  // Preallocate memory for the vector
  std::vector<std::vector<int>> vectorOfUvecs;
  vectorOfUvecs.reserve(std::accumulate(blocks_per_ranef.begin(), blocks_per_ranef.end(), 0));

  int values_ctr = 0;
  int total_ranef_coef_blocks_looped = 0;

  // loop over each random effect block
  for (int ranef_idx = 0; ranef_idx < terms_per_block.size(); ranef_idx++) {

    // loop over all the coefficients in the random effect block
    for (int j = 0; j < blocks_per_ranef[ranef_idx]; j++) {

      // Create the uvec equivalent in Eigen (std::vector<int>)
      std::vector<int> uvec(n_nz_terms[total_ranef_coef_blocks_looped]);

      for (int k = 0; k < n_nz_terms[total_ranef_coef_blocks_looped]; k++) {
        uvec[k] = static_cast<int>(Z_i[values_ctr]);
        values_ctr += 1;
      }

      vectorOfUvecs.push_back(uvec);
      values_ctr += (terms_per_block[ranef_idx] - 1) * n_nz_terms[total_ranef_coef_blocks_looped];
      total_ranef_coef_blocks_looped += 1;
    }
  }

  return vectorOfUvecs;
}


std::vector<Eigen::MatrixXd> initialize_S(
    const std::vector<int>& blocks_per_ranef,  // number of coefficients per ranef (e.g. 50)
    const std::vector<int>& terms_per_block
) {

  // Preallocate memory for the vector
  std::vector<Eigen::MatrixXd> vectorOfMats;
  int tot = std::accumulate(blocks_per_ranef.begin(), blocks_per_ranef.end(), 0);
  vectorOfMats.reserve(tot);

  // loop over each random effect block
  for (int k = 0; k < terms_per_block.size(); k++) {

    // loop over all the coefficients in the random effect block
    for (int j = 0; j < blocks_per_ranef[k]; j++) {

      vectorOfMats.push_back(
        Eigen::MatrixXd::Identity(terms_per_block[k], terms_per_block[k])
      );

    }
  }

  return vectorOfMats;
}

std::vector<int> log_chol_diag_idx_one_block(
  int terms
) {

  std::vector<int> idxVec;
  idxVec.reserve(terms);
  int running_rows_remaining = terms;

  int vals_iterated_through = 0;

  for (int i = 0; i < terms; i++) {

    idxVec.push_back(terms + vals_iterated_through);
    vals_iterated_through += running_rows_remaining;
    running_rows_remaining -= 1;

  }

  return idxVec;

}

std::vector<std::vector<int>> get_log_chol_diag_idx_per_ranef(
    const std::vector<int>& terms_per_block
) {

  // Preallocate memory for the vector
  std::vector<std::vector<int>> vectorOfIDX;
  vectorOfIDX.reserve(terms_per_block.size());

  // loop over each random effect block
  for (int k = 0; k < terms_per_block.size(); k++) {

    vectorOfIDX.push_back(
      log_chol_diag_idx_one_block(terms_per_block[k])
    );

  }

  return vectorOfIDX;
}

void create_Z_and_Z2(const std::vector<int>& Z_i,
                          const std::vector<int>& Z_j,
                          const std::vector<double>& Z_x,
                          Eigen::SparseMatrix<double>& Z,
                          Eigen::SparseMatrix<double>& Z2,
                          int rows, int cols) {
  // Number of non-zero elements
  int nnz = Z_x.size();

  // Reserve space for Z and Z2
  Z.resize(rows, cols);
  Z2.resize(rows, cols);

  // Create triplet lists for Z and Z2
  std::vector<Eigen::Triplet<double>> tripletListZ;
  std::vector<Eigen::Triplet<double>> tripletListZ2;

  // Fill triplets for Z and Z2
  for (int k = 0; k < nnz; ++k) {
    tripletListZ.emplace_back(Z_i[k], Z_j[k], Z_x[k]);
    tripletListZ2.emplace_back(Z_i[k], Z_j[k], Z_x[k] * Z_x[k]); // Z_x^2
  }

  // Set the triplets to the sparse matrices
  Z.setFromTriplets(tripletListZ.begin(), tripletListZ.end());
  Z2.setFromTriplets(tripletListZ2.begin(), tripletListZ2.end());
}

void printMatrix(const Eigen::MatrixXd& matrix) {
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      Rprintf("%f ", matrix(i, j));
    }
    Rprintf("\n");  // Newline after each row
  }
}

void printVector(const Eigen::VectorXd& vector) {
  for (int i = 0; i < vector.size(); ++i) {
    Rprintf("%f\n", vector(i));
  }
}
