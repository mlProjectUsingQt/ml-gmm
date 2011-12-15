close all;
clear all;
clc;

% mean1 = [1 2];
% mean2 = [-1 -2];
% covar1 = [3 0.2; 0.2 2];
% covar2 = [2 0; 0 1];
% trX = [mvnrnd(mean1, covar1, 200); mvnrnd(mean2, covar2, 100)];
% trX = [-6 0; -4 2; -5 -2; -3 0; 2 0; 3 2; 3 -2; 4 0];
% figure;
% scatter(trX(:, 1), trX(:, 2), 'ro', 'filled');
% grid on;

dat = load('mnist8vs9train.mat');
trX = dat.X;
N = length(trX);
K = floor(N / 100);
% K = 2;
[means Z] = gmm(trX, K);

[max_prob, hard_assgn] = max(Z, [], 2);
cluster1 = (hard_assgn == 1);
cluster2 = (hard_assgn == 2);
figure, hold on;
grid on;
scatter(trX(cluster1, 1), trX(cluster1, 2), 'r+');
scatter(trX(cluster2, 1), trX(cluster2, 2), 'bo');
scatter(means(1, 1), means(1, 2), 'ro', 'filled');
scatter(means(2, 1), means(2, 2), 'bo', 'filled');
hold off;