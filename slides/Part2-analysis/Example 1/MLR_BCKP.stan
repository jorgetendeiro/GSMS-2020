// Basic Stan model from here:
// https://mc-stan.org/docs/2_24/stan-users-guide/linear-regression.html

data {
  int<lower=0> N;   // number of data points
  int<lower=0> K;   // number of predictors
  matrix[N, K+1] x; // predictor matrix
  vector[N] y;      // outcome vector
}
parameters {
  vector[K+1] beta;     // intercept + regression coefficients
  real<lower=0> sigma;  // error scale
}
model {
  //y ~ normal(x * beta, sigma);  // likelihood
  target += normal_lpdf(y | x * beta, sigma); // useful for bridgesampling
}

generated quantities {
  vector[N] lin;
  vector[N] err;
  real<lower=0> R2;
  vector[N] y_ppc;
  vector[N] log_lik;
  real pred_age_70;
  
  // R squared:
  lin  = x * beta;
  err  = y - lin;
  R2   = variance(lin) / (variance(lin) + variance(err));
  
  //Posterior predictive checks:
  for (n in 1:N) {
    y_ppc[n] = normal_rng(lin[n], sigma);
  }
  
  // Log-likelihood:
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | lin[n], sigma);
  }
  
  // Out-of-sample prediction for subject with age = 70, group UD:
  for (n in 1:N) {
    pred_age_70 = normal_rng((beta[1] + beta[4]) + 70 * beta[5], sigma);
  }
}
