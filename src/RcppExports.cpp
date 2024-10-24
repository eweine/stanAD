// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// get_elbo_grad
Eigen::VectorXd get_elbo_grad(const Eigen::VectorXd& par_vals, const Eigen::VectorXd& Zty, const Eigen::VectorXd& Xty, const Eigen::MatrixXd& X, const Eigen::SparseMatrix<double>& Z, const Eigen::SparseMatrix<double>& Z2, const std::vector<int>& blocks_per_ranef, const std::vector<int>& terms_per_block, int& n_ranef_par, int& n_fixef_par);
RcppExport SEXP _stanAD_get_elbo_grad(SEXP par_valsSEXP, SEXP ZtySEXP, SEXP XtySEXP, SEXP XSEXP, SEXP ZSEXP, SEXP Z2SEXP, SEXP blocks_per_ranefSEXP, SEXP terms_per_blockSEXP, SEXP n_ranef_parSEXP, SEXP n_fixef_parSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type par_vals(par_valsSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Zty(ZtySEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Xty(XtySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<double>& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<double>& >::type Z2(Z2SEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type blocks_per_ranef(blocks_per_ranefSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type terms_per_block(terms_per_blockSEXP);
    Rcpp::traits::input_parameter< int& >::type n_ranef_par(n_ranef_parSEXP);
    Rcpp::traits::input_parameter< int& >::type n_fixef_par(n_fixef_parSEXP);
    rcpp_result_gen = Rcpp::wrap(get_elbo_grad(par_vals, Zty, Xty, X, Z, Z2, blocks_per_ranef, terms_per_block, n_ranef_par, n_fixef_par));
    return rcpp_result_gen;
END_RCPP
}
// get_elbo_hvp
Eigen::VectorXd get_elbo_hvp(const Eigen::VectorXd& par_vals, const Eigen::Matrix<double, Eigen::Dynamic, 1>& v, const Eigen::VectorXd& Zty, const Eigen::VectorXd& Xty, const Eigen::MatrixXd& X, const Eigen::SparseMatrix<double>& Z, const Eigen::SparseMatrix<double>& Z2, const std::vector<int>& blocks_per_ranef, const std::vector<int>& terms_per_block, int& n_ranef_par, int& n_fixef_par);
RcppExport SEXP _stanAD_get_elbo_hvp(SEXP par_valsSEXP, SEXP vSEXP, SEXP ZtySEXP, SEXP XtySEXP, SEXP XSEXP, SEXP ZSEXP, SEXP Z2SEXP, SEXP blocks_per_ranefSEXP, SEXP terms_per_blockSEXP, SEXP n_ranef_parSEXP, SEXP n_fixef_parSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type par_vals(par_valsSEXP);
    Rcpp::traits::input_parameter< const Eigen::Matrix<double, Eigen::Dynamic, 1>& >::type v(vSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Zty(ZtySEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Xty(XtySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<double>& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<double>& >::type Z2(Z2SEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type blocks_per_ranef(blocks_per_ranefSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type terms_per_block(terms_per_blockSEXP);
    Rcpp::traits::input_parameter< int& >::type n_ranef_par(n_ranef_parSEXP);
    Rcpp::traits::input_parameter< int& >::type n_fixef_par(n_fixef_parSEXP);
    rcpp_result_gen = Rcpp::wrap(get_elbo_hvp(par_vals, v, Zty, Xty, X, Z, Z2, blocks_per_ranef, terms_per_block, n_ranef_par, n_fixef_par));
    return rcpp_result_gen;
END_RCPP
}
// get_neg_elbo_pois_glmm
double get_neg_elbo_pois_glmm(Eigen::VectorXd& par, Eigen::VectorXd& par_scaling, Eigen::MatrixXd& X, std::vector<Eigen::MatrixXd>& vec_Z, std::vector<std::vector<int>>& y_nz_idx, const Eigen::VectorXd& Zty, const Eigen::VectorXd& Xty, const std::vector<int>& blocks_per_ranef, const std::vector<int>& log_chol_par_per_block, const std::vector<int>& terms_per_block, int& n, int& n_m_par, int& n_log_chol_par, int& n_b_par, int& total_blocks);
RcppExport SEXP _stanAD_get_neg_elbo_pois_glmm(SEXP parSEXP, SEXP par_scalingSEXP, SEXP XSEXP, SEXP vec_ZSEXP, SEXP y_nz_idxSEXP, SEXP ZtySEXP, SEXP XtySEXP, SEXP blocks_per_ranefSEXP, SEXP log_chol_par_per_blockSEXP, SEXP terms_per_blockSEXP, SEXP nSEXP, SEXP n_m_parSEXP, SEXP n_log_chol_parSEXP, SEXP n_b_parSEXP, SEXP total_blocksSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type par(parSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type par_scaling(par_scalingSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< std::vector<Eigen::MatrixXd>& >::type vec_Z(vec_ZSEXP);
    Rcpp::traits::input_parameter< std::vector<std::vector<int>>& >::type y_nz_idx(y_nz_idxSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Zty(ZtySEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Xty(XtySEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type blocks_per_ranef(blocks_per_ranefSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type log_chol_par_per_block(log_chol_par_per_blockSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type terms_per_block(terms_per_blockSEXP);
    Rcpp::traits::input_parameter< int& >::type n(nSEXP);
    Rcpp::traits::input_parameter< int& >::type n_m_par(n_m_parSEXP);
    Rcpp::traits::input_parameter< int& >::type n_log_chol_par(n_log_chol_parSEXP);
    Rcpp::traits::input_parameter< int& >::type n_b_par(n_b_parSEXP);
    Rcpp::traits::input_parameter< int& >::type total_blocks(total_blocksSEXP);
    rcpp_result_gen = Rcpp::wrap(get_neg_elbo_pois_glmm(par, par_scaling, X, vec_Z, y_nz_idx, Zty, Xty, blocks_per_ranef, log_chol_par_per_block, terms_per_block, n, n_m_par, n_log_chol_par, n_b_par, total_blocks));
    return rcpp_result_gen;
END_RCPP
}
// get_grad_pois_glmm
Eigen::VectorXd get_grad_pois_glmm(Eigen::VectorXd& par, Eigen::VectorXd& par_scaling, Eigen::MatrixXd& X, std::vector<Eigen::MatrixXd>& vec_Z, std::vector<std::vector<int>>& y_nz_idx, const Eigen::VectorXd& Zty, const Eigen::VectorXd& Xty, const std::vector<int>& blocks_per_ranef, const std::vector<int>& log_chol_par_per_block, const std::vector<int>& terms_per_block, int& n, int& n_m_par, int& n_log_chol_par, int& n_b_par, int& total_blocks);
RcppExport SEXP _stanAD_get_grad_pois_glmm(SEXP parSEXP, SEXP par_scalingSEXP, SEXP XSEXP, SEXP vec_ZSEXP, SEXP y_nz_idxSEXP, SEXP ZtySEXP, SEXP XtySEXP, SEXP blocks_per_ranefSEXP, SEXP log_chol_par_per_blockSEXP, SEXP terms_per_blockSEXP, SEXP nSEXP, SEXP n_m_parSEXP, SEXP n_log_chol_parSEXP, SEXP n_b_parSEXP, SEXP total_blocksSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type par(parSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type par_scaling(par_scalingSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< std::vector<Eigen::MatrixXd>& >::type vec_Z(vec_ZSEXP);
    Rcpp::traits::input_parameter< std::vector<std::vector<int>>& >::type y_nz_idx(y_nz_idxSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Zty(ZtySEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Xty(XtySEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type blocks_per_ranef(blocks_per_ranefSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type log_chol_par_per_block(log_chol_par_per_blockSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type terms_per_block(terms_per_blockSEXP);
    Rcpp::traits::input_parameter< int& >::type n(nSEXP);
    Rcpp::traits::input_parameter< int& >::type n_m_par(n_m_parSEXP);
    Rcpp::traits::input_parameter< int& >::type n_log_chol_par(n_log_chol_parSEXP);
    Rcpp::traits::input_parameter< int& >::type n_b_par(n_b_parSEXP);
    Rcpp::traits::input_parameter< int& >::type total_blocks(total_blocksSEXP);
    rcpp_result_gen = Rcpp::wrap(get_grad_pois_glmm(par, par_scaling, X, vec_Z, y_nz_idx, Zty, Xty, blocks_per_ranef, log_chol_par_per_block, terms_per_block, n, n_m_par, n_log_chol_par, n_b_par, total_blocks));
    return rcpp_result_gen;
END_RCPP
}
// get_elbo_pois_glmm_MFVB
double get_elbo_pois_glmm_MFVB(const Eigen::VectorXd& par_vals, const Eigen::VectorXd& Zty, const Eigen::VectorXd& Xty, const Eigen::MatrixXd& X, const Eigen::SparseMatrix<double>& Z, const Eigen::SparseMatrix<double>& Z2, const std::vector<int>& blocks_per_ranef, const std::vector<int>& terms_per_block, int& n_ranef_par, int& n_fixef_par);
RcppExport SEXP _stanAD_get_elbo_pois_glmm_MFVB(SEXP par_valsSEXP, SEXP ZtySEXP, SEXP XtySEXP, SEXP XSEXP, SEXP ZSEXP, SEXP Z2SEXP, SEXP blocks_per_ranefSEXP, SEXP terms_per_blockSEXP, SEXP n_ranef_parSEXP, SEXP n_fixef_parSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type par_vals(par_valsSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Zty(ZtySEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Xty(XtySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<double>& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<double>& >::type Z2(Z2SEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type blocks_per_ranef(blocks_per_ranefSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type terms_per_block(terms_per_blockSEXP);
    Rcpp::traits::input_parameter< int& >::type n_ranef_par(n_ranef_parSEXP);
    Rcpp::traits::input_parameter< int& >::type n_fixef_par(n_fixef_parSEXP);
    rcpp_result_gen = Rcpp::wrap(get_elbo_pois_glmm_MFVB(par_vals, Zty, Xty, X, Z, Z2, blocks_per_ranef, terms_per_block, n_ranef_par, n_fixef_par));
    return rcpp_result_gen;
END_RCPP
}
// fit_pois_glmm_block_posterior_ccd
Rcpp::List fit_pois_glmm_block_posterior_ccd(Eigen::VectorXd& m, Eigen::VectorXd& S_log_chol, Eigen::VectorXd& b, Eigen::VectorXd& link_offset, const std::vector<int>& n_nz_terms_per_col, const std::vector<int>& terms_per_block, const std::vector<int>& blocks_per_ranef, const std::vector<int>& log_chol_par_per_block, const Eigen::VectorXd& Zty, const Eigen::VectorXd& Xty, const Eigen::MatrixXd& X, const std::vector<int>& Z_i, const std::vector<int>& Z_j, const std::vector<double>& Z_x, Eigen::SparseMatrix<double>& Z, Eigen::SparseMatrix<double>& Z2, double elbo_tol, const int& num_iter, const bool is_mfvb);
RcppExport SEXP _stanAD_fit_pois_glmm_block_posterior_ccd(SEXP mSEXP, SEXP S_log_cholSEXP, SEXP bSEXP, SEXP link_offsetSEXP, SEXP n_nz_terms_per_colSEXP, SEXP terms_per_blockSEXP, SEXP blocks_per_ranefSEXP, SEXP log_chol_par_per_blockSEXP, SEXP ZtySEXP, SEXP XtySEXP, SEXP XSEXP, SEXP Z_iSEXP, SEXP Z_jSEXP, SEXP Z_xSEXP, SEXP ZSEXP, SEXP Z2SEXP, SEXP elbo_tolSEXP, SEXP num_iterSEXP, SEXP is_mfvbSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type m(mSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type S_log_chol(S_log_cholSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type b(bSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type link_offset(link_offsetSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type n_nz_terms_per_col(n_nz_terms_per_colSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type terms_per_block(terms_per_blockSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type blocks_per_ranef(blocks_per_ranefSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type log_chol_par_per_block(log_chol_par_per_blockSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Zty(ZtySEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type Xty(XtySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type Z_i(Z_iSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type Z_j(Z_jSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type Z_x(Z_xSEXP);
    Rcpp::traits::input_parameter< Eigen::SparseMatrix<double>& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< Eigen::SparseMatrix<double>& >::type Z2(Z2SEXP);
    Rcpp::traits::input_parameter< double >::type elbo_tol(elbo_tolSEXP);
    Rcpp::traits::input_parameter< const int& >::type num_iter(num_iterSEXP);
    Rcpp::traits::input_parameter< const bool >::type is_mfvb(is_mfvbSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_pois_glmm_block_posterior_ccd(m, S_log_chol, b, link_offset, n_nz_terms_per_col, terms_per_block, blocks_per_ranef, log_chol_par_per_block, Zty, Xty, X, Z_i, Z_j, Z_x, Z, Z2, elbo_tol, num_iter, is_mfvb));
    return rcpp_result_gen;
END_RCPP
}
// H
Eigen::MatrixXd H(Eigen::VectorXd x, Eigen::VectorXd a);
RcppExport SEXP _stanAD_H(SEXP xSEXP, SEXP aSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type a(aSEXP);
    rcpp_result_gen = Rcpp::wrap(H(x, a));
    return rcpp_result_gen;
END_RCPP
}
// log_cholesky_grad
Eigen::VectorXd log_cholesky_grad(const Eigen::VectorXd& l_params, const Eigen::MatrixXd& A);
RcppExport SEXP _stanAD_log_cholesky_grad(SEXP l_paramsSEXP, SEXP ASEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type l_params(l_paramsSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type A(ASEXP);
    rcpp_result_gen = Rcpp::wrap(log_cholesky_grad(l_params, A));
    return rcpp_result_gen;
END_RCPP
}
// get_n_nz_terms
std::vector<int> get_n_nz_terms(const std::vector<int> n_nz_terms_per_col, const std::vector<int> blocks_per_ranef, const std::vector<int> terms_per_block);
RcppExport SEXP _stanAD_get_n_nz_terms(SEXP n_nz_terms_per_colSEXP, SEXP blocks_per_ranefSEXP, SEXP terms_per_blockSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<int> >::type n_nz_terms_per_col(n_nz_terms_per_colSEXP);
    Rcpp::traits::input_parameter< const std::vector<int> >::type blocks_per_ranef(blocks_per_ranefSEXP);
    Rcpp::traits::input_parameter< const std::vector<int> >::type terms_per_block(terms_per_blockSEXP);
    rcpp_result_gen = Rcpp::wrap(get_n_nz_terms(n_nz_terms_per_col, blocks_per_ranef, terms_per_block));
    return rcpp_result_gen;
END_RCPP
}
// create_ranef_Z
std::vector<Eigen::MatrixXd> create_ranef_Z(const std::vector<int>& n_nz_terms, const std::vector<int>& blocks_per_ranef, const std::vector<int>& terms_per_block, const std::vector<double>& values);
RcppExport SEXP _stanAD_create_ranef_Z(SEXP n_nz_termsSEXP, SEXP blocks_per_ranefSEXP, SEXP terms_per_blockSEXP, SEXP valuesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<int>& >::type n_nz_terms(n_nz_termsSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type blocks_per_ranef(blocks_per_ranefSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type terms_per_block(terms_per_blockSEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type values(valuesSEXP);
    rcpp_result_gen = Rcpp::wrap(create_ranef_Z(n_nz_terms, blocks_per_ranef, terms_per_block, values));
    return rcpp_result_gen;
END_RCPP
}
// create_y_nz_idx
std::vector<std::vector<int>> create_y_nz_idx(const std::vector<int>& n_nz_terms, const std::vector<int>& blocks_per_ranef, const std::vector<int>& terms_per_block, const std::vector<int>& Z_i);
RcppExport SEXP _stanAD_create_y_nz_idx(SEXP n_nz_termsSEXP, SEXP blocks_per_ranefSEXP, SEXP terms_per_blockSEXP, SEXP Z_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<int>& >::type n_nz_terms(n_nz_termsSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type blocks_per_ranef(blocks_per_ranefSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type terms_per_block(terms_per_blockSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type Z_i(Z_iSEXP);
    rcpp_result_gen = Rcpp::wrap(create_y_nz_idx(n_nz_terms, blocks_per_ranef, terms_per_block, Z_i));
    return rcpp_result_gen;
END_RCPP
}
// structure_output
Rcpp::List structure_output(const Eigen::VectorXd& par, const std::vector<int>& blocks_per_ranef, const std::vector<int>& log_chol_par_per_block, const std::vector<int>& terms_per_block, int& n_m_par, int& n_log_chol_par, int& n_b_par, int& total_blocks);
RcppExport SEXP _stanAD_structure_output(SEXP parSEXP, SEXP blocks_per_ranefSEXP, SEXP log_chol_par_per_blockSEXP, SEXP terms_per_blockSEXP, SEXP n_m_parSEXP, SEXP n_log_chol_parSEXP, SEXP n_b_parSEXP, SEXP total_blocksSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type par(parSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type blocks_per_ranef(blocks_per_ranefSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type log_chol_par_per_block(log_chol_par_per_blockSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type terms_per_block(terms_per_blockSEXP);
    Rcpp::traits::input_parameter< int& >::type n_m_par(n_m_parSEXP);
    Rcpp::traits::input_parameter< int& >::type n_log_chol_par(n_log_chol_parSEXP);
    Rcpp::traits::input_parameter< int& >::type n_b_par(n_b_parSEXP);
    Rcpp::traits::input_parameter< int& >::type total_blocks(total_blocksSEXP);
    rcpp_result_gen = Rcpp::wrap(structure_output(par, blocks_per_ranef, log_chol_par_per_block, terms_per_block, n_m_par, n_log_chol_par, n_b_par, total_blocks));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_stanAD_get_elbo_grad", (DL_FUNC) &_stanAD_get_elbo_grad, 10},
    {"_stanAD_get_elbo_hvp", (DL_FUNC) &_stanAD_get_elbo_hvp, 11},
    {"_stanAD_get_neg_elbo_pois_glmm", (DL_FUNC) &_stanAD_get_neg_elbo_pois_glmm, 15},
    {"_stanAD_get_grad_pois_glmm", (DL_FUNC) &_stanAD_get_grad_pois_glmm, 15},
    {"_stanAD_get_elbo_pois_glmm_MFVB", (DL_FUNC) &_stanAD_get_elbo_pois_glmm_MFVB, 10},
    {"_stanAD_fit_pois_glmm_block_posterior_ccd", (DL_FUNC) &_stanAD_fit_pois_glmm_block_posterior_ccd, 19},
    {"_stanAD_H", (DL_FUNC) &_stanAD_H, 2},
    {"_stanAD_log_cholesky_grad", (DL_FUNC) &_stanAD_log_cholesky_grad, 2},
    {"_stanAD_get_n_nz_terms", (DL_FUNC) &_stanAD_get_n_nz_terms, 3},
    {"_stanAD_create_ranef_Z", (DL_FUNC) &_stanAD_create_ranef_Z, 4},
    {"_stanAD_create_y_nz_idx", (DL_FUNC) &_stanAD_create_y_nz_idx, 4},
    {"_stanAD_structure_output", (DL_FUNC) &_stanAD_structure_output, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_stanAD(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
