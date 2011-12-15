function [means Z] = gmm (X, K)
% means: K * D
% Z: N * K
% X: N * D
% K: scalar
means = gmm_init(X, K, 'random'); % Initialization
max_itr = 20; % Max # of iterations
loglik_th = 1e-6; % log likelihood difference threshold
loglik_prev = -1.0; % last iteration log likelihood
[N D] = size(X);
covars = zeros(K, D);
weights = ones(K, 1) / K;
[probs diffs] = gaussian(X, means, covars);
wprobs = probs .* repmat(weights', N, 1);
wprobs_sum = sum(wprobs, 2);
for itr = 1: max_itr
  % E-step
  Z = wprobs ./ repmat(wprobs_sum, 1, K);
  % M-step
  Nk = sum(Z, 1)';
  means = Z' * X ./repmat(Nk, 1, D);
  for k = 1: K
    covars(k, :) = Z(:, k)' * (squeeze(diffs(:, k, :)) .^ 2) / Nk(k);
  end
  weights = Nk / N;
  % Update distributions
  [probs diffs] = gaussian(X, means, covars);
  wprobs = probs .* repmat(weights', N, 1);
  wprobs_sum = sum(wprobs, 2);
  % Check convergence
  loglik = sum(log(wprobs_sum));
  if abs(loglik - loglik_prev) < loglik_th
    break;
  end
  loglik_prev = loglik;
end



function [probs diffs] = gaussian (X, means, covars)
% probs: N * K
% diffs: N * K * D
% X: N * D
% means: K * D
% covars: K * D
[N D] = size(X);
K = size(means, 1);
diffs = zeros(N, K, D);
expo = zeros(N, K); % N * K
for d = 1: D
  diffs(:, :, d) = repmat(X(:, d), 1, K) - repmat(means(:, d)', N, 1);
  expo = expo + ...
    squeeze(diffs(:, :, d)) .^ 2 ./ repmat(covars(:, d)', N, 1);
end
probs = exp(-(repmat((log(prod(covars, 2)))', N, 1) + expo) / 2);



function means = gmm_init (X, K, init_type)
% means: K * D
% covars: K * D
% weights: K * 1
% X: N * D
% K: scalar
% init_type: string
N = size(X, 1);
if nargin < 3 || strcmp(init_type, 'random')
  perm = randperm(N);
  perm = perm(1: K);
  means = X(perm, :);
end