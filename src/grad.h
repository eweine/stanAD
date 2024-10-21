#ifndef GRAD_H
#define GRAD_H

#include <RcppEigen.h>

Eigen::VectorXd single_local_block_multiD_grad_pois_glmm(
    const Eigen::VectorXd Zty,
    const Eigen::MatrixXd& Z,
    Eigen::VectorXd par_vals, // parameters in the order (m1, m2, ls1, ls2, ls3)
    Eigen::VectorXd par_scaling,
    Eigen::MatrixXd& Sigma_inv,
    Eigen::VectorXd link,
    int n_m_par
);

Eigen::VectorXd single_local_block_1D_grad_pois_glmm(
    const double Zty,
    const Eigen::VectorXd& z,
    const Eigen::VectorXd& z2,
    Eigen::Vector<double, 2> par_vals, // parameters in the order (m1, ls1)
    Eigen::Vector<double, 2> par_scaling,
    double sigma2,
    Eigen::VectorXd& link
);

double single_var_comp_1D_grad_glmm(
    Eigen::VectorXd& m,
    Eigen::VectorXd& s2,
    double par_scaling,
    double log_sigma
);

Eigen::VectorXd single_var_comp_multiD_grad_glmm(
    Eigen::MatrixXd& M,
    Eigen::MatrixXd& S,
    Eigen::VectorXd par_scaling,
    Eigen::VectorXd Sigma_log_chol,
    int Sigma_d
);

Eigen::VectorXd fixef_grad_pois_glmm(
    const Eigen::VectorXd& Xty,
    const Eigen::MatrixXd& X,
    Eigen::VectorXd b,
    Eigen::VectorXd b_scaling,
    Eigen::VectorXd link
);

#endif
