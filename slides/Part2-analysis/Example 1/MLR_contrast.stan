// Basic Stan model from here:
// https://mc-stan.org/docs/2_24/stan-users-guide/linear-regression.html

// This model is based on H0: mean(Control) = mean(UD)
data {
  int<lower=0> N;   // number of data points
  int<lower=0> K;   // number of predictors
  matrix[N, K+1] x; // predictor matrix
  vector[N] y;      // outcome variable
}

parameters {
  vector[K] beta0;      // intercept + (unique) regression coefficients
  real<lower=0> sigma;  // SD residuals
}

transformed parameters {
  vector[K+1] beta;
  beta[1] = beta0[1];
  beta[2] = beta0[2];
  beta[3] = beta0[3];
  beta[4] = beta0[3];
  beta[5] = beta0[4];
}

model {
  // Adapted for bridesampling. See here: https://arxiv.org/pdf/1710.08162.pdf
  target += normal_lpdf(beta | 0, 10);                              // prior beta
  target += cauchy_lpdf(sigma | 0, 1) - 1 * cauchy_lccdf(0 | 0, 1); // prior sigma
  target += normal_lpdf(y | x * beta, sigma);                       // likelihood
}

generated quantities {
  vector[N] lin;
  vector[N] err;
  real<lower=0> R2;
  vector[N] y_ppd;
  vector[N] log_lik;
  
  // R squared:
  lin  = x * beta;
  err  = y - lin;
  R2   = variance(lin) / (variance(lin) + variance(err));
  
  //Posterior predictive checks:
  for (n in 1:N) {
   y_ppd[n] = normal_rng(lin[n], sigma);
  }
  
  // Log-likelihood:
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | lin[n], sigma);
  }
}
