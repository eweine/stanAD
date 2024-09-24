data {
  int<lower=1> N;                  // Number of data points
  int<lower=1> K;                  // Number of predictors
  matrix[N, K] X;                  // Design matrix (predictors)
  int<lower=0> y[N];               // Response variable (counts)
  cov_matrix[K] Sigma_prior;       // Prior covariance matrix for coefficients
}

parameters {
  vector[K] beta;                  // Regression coefficients
}

model {
  // Priors
  beta ~ multi_normal(rep_vector(0, K), Sigma_prior); // Custom multivariate normal prior

  // Likelihood
  y ~ poisson_log(X * beta);       // Poisson likelihood with log-link
}

