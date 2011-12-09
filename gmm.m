function [C Z] = gmm (X, K)
% Use EM to fit Gaussian mixture model
% C: K-by-D matrix, cluster centers
% Z: N-by-K matrix, soft assignments
% X: N-by-D matrix, data points
% K: scalar, # of clusters
max_iteration = 100;
loglikelihood_threshold = 1e-6;
prev_loglikelihood = -1.0;
[N D] = size(X);
% Initialization
% CV: K-by-D-by-D matrix, covariances for each cluster
% weights: K-by-1 vector, weight (pi_k) for each cluster
[C CV weights] = gmm_init(X, K);
weighted_gaussians = gaussian(X, C, CV) .* repmat(weights, 1, N);
sum_over_k_weighted_gaussians = sum(weighted_gaussians, 2);
for itr = 1: max_iteration
  % E-step
  % Evaluate responsibilities (PRML: eq. 9.23)
  Z = weighted_gaussians ./ repmat(sum_over_k_weighted_gaussians, 1, K);
  % M-step
  % Compute Nk (K-by-1 vector) (PRML: eq. 9.27)
  Nk = sum(Z, 1)';
  % Estimate means (PRML: eq. 9.24)
  C = Z' * X ./repmat(Nk, 1, D);
  % Estimate covariances (PRML: eq. 9.25)
  for k = 1: K
    diff = X - repmat(C(k, :), N, 1);
    CV(k, :, :) = repmat(Z(:, k)', D, 1) .* diff' * diff;
  end
  % Estimate weights (PRML: eq. 9.26)
  weights = Nk / N;
  % Update weighted gaussians and its sum-over-k
  weighted_gaussians = gaussian(X, C, CV) .* repmat(weights, 1, N);
  sum_over_k_weighted_gaussians = sum(weighted_gaussians, 2);
  % Evaluate log likelihood
  loglikelihood = sum(ln(sum_over_k_weighted_gaussians));
  % Check convergence
  if abs(loglikelihood - prev_loglikelihood) < loglikelihood_threshold
    break;
  end
  prev_loglikelihood = loglikelihood;
end

function prob = gaussian (X, means, covars)
% Compute probabilities w.r.t. Gaussian distribution
% prob: N-by-K matrix, prob(n, k) = Nor(X(N, :), means(k, :), covars(k, :, :))
% X: N-by-D matrix, data points
% means: K-by-D matrix, means
% covars: K-by-D-by-D matrix, covariances
[N D] = size(X);
K = size(means, 1);
prob = zeros(N, K);
coef = 1.0 / (2.0 * pi) ^ (D / 2.0);
for k = 1: K
  diff = X - repmat(means(k, :), N, 1);
  cv = squeeze(covars(k, :, :));
  prob(:, k) = exp(-0.5 * diff' / cv * diff) / sqrt(det(cv)) * coef;
end

function [C CV weights] = gmm_init (X, K)
% GMM initialization
% C: K-by-D matrix, cluster centers
% CV: K-by-D-by-D matrix, covariances for each cluster
% weights: K-by-1 vector, weight (pi_k) for each cluster
% X: N-by-D matrix, data points
% K: scalar, # of clusters
[N D] = size(X);
perm = randperm(N);
perm = perm(1: K);
C = X(perm, :);
Z = zeros(N, 1);