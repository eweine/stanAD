#include <RcppEigen.h>
#include <functional>

// Modified solve_cg to accept std::function instead of a function pointer
Eigen::VectorXd solve_cg(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> hvp_func,
    Eigen::VectorXd& x,
    Eigen::VectorXd& b,
    double tol
) {
  Eigen::VectorXd Hv = hvp_func(x);
  Eigen::VectorXd r = b - Hv;

  Eigen::VectorXd p = r;
  double a;
  double beta;

  double r_dot_r = r.dot(r);
  double new_r_dot_r;

  int cg_iter = 0;

  while (true) {
    Hv = hvp_func(p);
    a = r_dot_r / (p.dot(Hv));
    x += a * p;
    r -= a * Hv;

    if (r.norm() < tol) {
      Rprintf("Took %i cg iterations\n", cg_iter);
      break;
    }

    new_r_dot_r = r.dot(r);
    beta = new_r_dot_r / r_dot_r;
    r_dot_r = new_r_dot_r;
    p = r + beta * p;

    cg_iter += 1;

  }

  return x;
}

Eigen::VectorXd solve_cg_diag_precond(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> hvp_func,
    Eigen::VectorXd& x,
    Eigen::VectorXd& b,
    const Eigen::DiagonalMatrix<double, Eigen::Dynamic>& M_inv,
    double tol
) {
  Eigen::VectorXd Hv = hvp_func(x);
  Eigen::VectorXd r = b - Hv;
  Eigen::VectorXd z = M_inv * r;

  Eigen::VectorXd p = z;
  double a;
  double beta;

  double r_dot_z = r.dot(z);
  double new_r_dot_z;

  int cg_iter = 0;

  while (true) {
    Hv = hvp_func(p);
    a = r_dot_z / (p.dot(Hv));
    x += a * p;
    r -= a * Hv;

    if (r.norm() < tol) {
      Rprintf("Took %i cg iterations\n", cg_iter);
      break;
    }

    z = M_inv * r;
    new_r_dot_z = r.dot(z);
    beta = new_r_dot_z / r_dot_z;
    r_dot_z = new_r_dot_z;
    p = z + beta * p;

    cg_iter += 1;

  }

  return x;
}

//
// Eigen::VectorXd test_solve_cg(
//     const Eigen::MatrixXd& A,  // Matrix should be passed by const reference
//     Eigen::VectorXd b,
//     Eigen::VectorXd x0
// ) {
//   // Define the lambda that computes A * x
//   auto get_hvp = [&A](const Eigen::VectorXd& x) {
//     return A * x;
//   };
//
//   // Set a tolerance value for the CG solver
//   double tol = 1e-6;
//
//   // Call solve_cg with the lambda
//   Eigen::VectorXd sol = solve_cg(get_hvp, x0, b, tol);
//
//   return sol;
// }
