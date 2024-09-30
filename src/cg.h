#ifndef CG_H
#define CG_H

#include <RcppEigen.h>
#include <functional>

Eigen::VectorXd solve_cg(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> hvp_func,
    Eigen::VectorXd& x,
    Eigen::VectorXd& b,
    double tol
);

#endif
