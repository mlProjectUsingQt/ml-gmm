function [L W I] = gmm_weight (K, Z, D)
% Compute labels and weights for cluster centers
% L: K-by-1 vector, labels for cluster centers
% W: K-by-1 vecotr, weights for 
% I: N-by-1 vector, hard assignments
% K: scalar, # of clusters
% Z: N-by-K matrix, soft assignments
% D: N-by-1 vector, truth labels
LP = unique(D); % Pool of labels
M = length(LP); % # of unique labels
L = zeros(K, 1);
W = zeros(K, 1);
for k = 1: K
  l = zeros(M, 1);
  for m = 1: M
    l(m) = sum(Z(D == LP(m), k));
  end
  [maxi idx] = max(l);
  L(k) = LP(idx);
  W(k) = sum(Z(:, k));
end
W = W / sum(W);
[maxi I] = max(Z, [], 2);