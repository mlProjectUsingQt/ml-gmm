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
% CV: D-by-D-by-K matrix, covariances for each cluster
% weights: K-by-1 vector, weight (pi_k) for each cluster
gmm_init_start_time = tic;
[C CV weights] = gmm_init(X, K);
gmm_init_time = toc(gmm_init_start_time);
fprintf('GMM init time (s): %f\n', gmm_init_time);
weighted_gaussians = gaussian(X, C, CV) .* repmat(weights', N, 1);
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
    CV(:, :, k) = (repmat(Z(:, k)', D, 1) .* diff') * diff / Nk(k);
  end
  % Estimate weights (PRML: eq. 9.26)
  weights = Nk / N;
  
%   [max_porb hard_assignment] = max(Z, [], 2);
%   cluster1 = (hard_assignment == 1);
%   cluster2 = (hard_assignment == 2);
%   close;
%   figure, grid on;
%   hold on;
%   scatter(X(cluster1, 1), X(cluster1, 2), 'r+');
%   scatter(X(cluster2, 1), X(cluster2, 2), 'bo');
%   scatter(C(1, 1), C(1, 2), 'ro', 'filled');
%   scatter(C(2, 1), C(2, 2), 'bo', 'filled');
%   hold off;
%   pause();
  
  % Update weighted gaussians and its sum-over-k
  weighted_gaussians = gaussian(X, C, CV) .* repmat(weights', N, 1);
  sum_over_k_weighted_gaussians = sum(weighted_gaussians, 2);
  % Evaluate log likelihood
  loglikelihood = sum(log(sum_over_k_weighted_gaussians));
  % Check convergence
  if abs(loglikelihood - prev_loglikelihood) < loglikelihood_threshold
    break;
  end
  prev_loglikelihood = loglikelihood;
end
% disp([num2str(itr) ' iterations, log-likelihood = ' ...
%     num2str(loglikelihood)]);

function prob = gaussian (X, means, covars)
% Compute probabilities w.r.t. Gaussian distribution
% prob: N-by-K matrix, prob(n, k) = Nor(X(N, :), means(k, :), covars(k, :, :))
% X: N-by-D matrix, data points
% means: K-by-D matrix, means
% covars: D-by-D-by-K matrix, covariances
% [N D] = size(X);
N = size(X, 1);
K = size(means, 1);
prob = zeros(N, K);
for k = 1: K
  prob(:, k) = mvnpdf(X, means(k, :), squeeze(covars(:, :, k)));
%   sigma = squeeze(covars(:, :, k));
%   mu = means(k, :);
%   coef = (2 * pi) ^ (-D / 2) * det(sigma) ^ (-0.5);
%   parfor n = 1: N
%     diff = X(n, :) - mu;
%     prob(n, k) = exp(-0.5 * diff * sigma * diff') * coef;
%   end
end

function [C CV weights] = gmm_init (X, K)
% Use k-means to initialize GMM
% C: K-by-D matrix, cluster centers
% CV: D-by-D-by-K matrix, covariances for each cluster
% weights: K-by-1 vector, weight (pi_k) for each cluster
% X: N-by-D matrix, data points
% K: scalar, # of clusters

% opts = statset('MaxIter', 10);
% [idx C] = kmeans(X, K, 'start', 'sample', 'options', opts);
% [N D] = size(X);
% CV = zeros(D, D, K);
% weights = zeros(K, 1);
% for k = 1: K
%   Xk = X(idx == k, :);
%   Nk = size(Xk, 1);
%   diff = Xk - repmat(C(k, :), Nk, 1);
%   CV(:, :, k) = diff' * diff / Nk;
%   weights(k) = Nk / N;
% end

[N D] = size(X);
C = zeros(K, D);
isPicked = zeros(N, 1);
idx = randi(N);
C(1, :) = X(idx, :);
isPicked(idx) = 1;
for k = 2: K
  C_bar = mean(C(1: k - 1, :), 1);
  dist2 = sum((X - repmat(C_bar, N, 1)) .^ 2, 2);
  dmax = -1.0;
  idx = 0;
  for n = 1: N
    if isPicked(n) == 0 && dist2(n) > dmax
      dmax = dist2(n);
      idx = n;
    end
  end
  C(k, :) = X(idx, :);
  isPicked(idx) = 1;
end
Z = zeros(N, 1);
for n = 1: N
  d2 = sum((repmat(X(n, :), K, 1) - C) .^ 2, 2);
  [dmax k] = min(d2);
  Z(n) = k;
end
CV = zeros(D, D, K);
weights = zeros(K, 1);
for k = 1: K
  Xk = X(Z == k, :);
  Nk = size(Xk, 1);
  diff = Xk - repmat(C(k, :), Nk, 1);
  CV(:, :, k) = diff' * diff / Nk;
  weights(k) = Nk / N;
end
% cluster1 = (idx == 1);
% cluster2 = (idx == 2);
% figure, hold on;
% grid on;
% scatter(X(cluster1, 1), X(cluster1, 2), 'r+');
% scatter(X(cluster2, 1), X(cluster2, 2), 'bo');
% scatter(C(1, 1), C(1, 2), 'ro', 'filled');
% scatter(C(2, 1), C(2, 2), 'bo', 'filled');