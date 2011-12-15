function [means Z] = gmm (X, K)
% means: K * D, cluster centers
% Z: N * K, responsibilities
% X: N * D, data points
% K: scalar, # of clusters
[means covars weights] = gmm_init(X, K, 'random'); % Initialization
max_itr = 20; % Max # of iterations
loglik_th = 1e-6; % log likelihood difference threshold
loglik_prev = -1.0; % last iteration log likelihood
[N D] = size(X);
[probs diffs] = gaussian(X, means, covars); % Update distributions
wprobs = probs .* repmat(weights', N, 1); % Weight distributions
wprobs_sum = sum(wprobs, 2); % Sum over k
for itr = 1: max_itr
  % E-step
  Z = wprobs ./ repmat(wprobs_sum, 1, K); % Update responsibilities
  % M-step
  Nk = sum(Z, 1)';
  means = Z' * X ./repmat(Nk, 1, D); % Update cluster centers
  % Update covariances
  for k = 1: K
    covars(k, :) = Z(:, k)' * (squeeze(diffs(:, k, :)) .^ 2) / Nk(k);
  end
  weights = Nk / N; % Update weights
  [probs diffs] = gaussian(X, means, covars); % Update distributions
  wprobs = probs .* repmat(weights', N, 1); % Weight distributions
  wprobs_sum = sum(wprobs, 2); % Sum over k
  % Check convergence
  loglik = sum(log(wprobs_sum));
  if abs(loglik - loglik_prev) < loglik_th
    break;
  end
  loglik_prev = loglik; % Keep track of last log likelihood
end



function [probs diffs] = gaussian (X, means, covars)
% probs: N * K, Gaussian distributions
% diffs: N * K * D, diff(n, k) = X(n, :) - means(k, :)
% X: N * D, data points
% means: K * D, cluster centers
% covars: K * D, covariances
[N D] = size(X);
K = size(means, 1);
diffs = zeros(N, K, D);
expo = zeros(N, K); % N * K, exponents
for d = 1: D % Compute for each dimension and add up
  diffs(:, :, d) = repmat(X(:, d), 1, K) - repmat(means(:, d)', N, 1);
  expo = expo + ...
    squeeze(diffs(:, :, d)) .^ 2 ./ repmat(covars(:, d)', N, 1);
end
probs = exp(-(repmat((log(prod(covars, 2)))', N, 1) + expo) / 2);



function [means covars weights] = gmm_init (X, K, init_type)
% means: K * D
% covars: K * D
% weights: K * 1
% X: N * D
% K: scalar
% init_type: string
[N D] = size(X);
covars = zeros(K, D);
if nargin < 3 || strcmp(init_type, 'random') % Randomly pick centers
  perm = randperm(N);
  means = X(perm(1: K), :);
  weights = ones(K, 1) / K; % Each point is weighted equally
  covars = ones(K, D); % Trivially initialize covariances
elseif strcmp(init_type, 'kmeans') % Run kmeans for centers
  [idx means] = kmeans(X, K);
  covars = zeros(K, D);
  weights = zeros(K, 1);
  for k = 1: K
    Xk = X(idx == k, :);
    Nk = size(Xk, 1);
    covars(k, :) = sum((Xk - repmat(means(k, :), Nk, 1)) .^ 2, 1);
    weights(k) = Nk / N;
  end
end