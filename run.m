close all;
clear all;
clc;

mu1 = [1.0 2.0];
mu2 = [-1.0 -2.0];
sigma1 = [3.0 0.2; 0.2 2.0];
sigma2 = [2.0 0.0; 0.0 1.0];
X = [mvnrnd(mu1, sigma1, 200); mvnrnd(mu2, sigma2, 100)];
scatter(X(:, 1), X(:, 2), 10, 'ko');

[C Z] = gmm(X, 2);
