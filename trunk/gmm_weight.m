function [L W I] = gmm_weight (C, Z, D)
% Compute labels and weights for cluster centers
% L: K-by-1 vector, labels for cluster centers
% W: K-by-1 vecotr, weights for 
% I: N-by-1 vector, hard assignments
% C: K-by-D matrix, cluster centers
% Z: N-by-K matrix, soft assignments
% D: N-by-1 vector, truth labels
K = size(C, 1);
LP = unique(D); % Pool of labels
M = length(P); % # of unique labels
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