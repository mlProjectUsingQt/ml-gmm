close all;
clear all;
clc;

mean1 = [1.0 2.0];
mean2 = [-1.0 -2.0];
covar1 = [3.0 0.2; 0.2 2.0];
covar2 = [2.0 0.0; 0.0 1.0];
X = [mvnrnd(mean1, covar1, 200); mvnrnd(mean2, covar2, 100)];
% scatter(X(:, 1), X(:, 2), 10, 'ko');
% X = [-2 -2; -1 -2; 0 -2; -2 -1; 0 -1; -2 0; -1 0; ...
%   1 0; 2 0; 0 1; 1 1; 2 1; 0 2; 1 2; 2 2];
% X = [-2 -2; -1 -2; -2 -1; ...
% X = [3 2; 2 0; 4 0; 3 -2; ...
%   -4 2; -6 0; -3 0; -5 -2];
%   1 0; 2 0; 0 1; 1 1; 2 1; 0 2; 1 2; 2 2];
% figure, hold on;
% grid on;
% scatter(X(:, 1), X(:, 2), 10, 'ko');

options = statset('Display', 'final');
gm = gmdistribution.fit(X, 2, 'Options', options);
figure, hold on;
ezcontour(@(x, y)pdf(gm, [x y]));
scatter(X(:, 1), X(:, 2), 'bo');
grid on;
hold off;

[C Z] = gmm(X, 2);

[max_porb hard_assignment] = max(Z, [], 2);
cluster1 = (hard_assignment == 1);
cluster2 = (hard_assignment == 2);
figure, hold on;
grid on;
scatter(X(cluster1, 1), X(cluster1, 2), 'r+');
scatter(X(cluster2, 1), X(cluster2, 2), 'bo');
scatter(C(1, 1), C(1, 2), 'ro', 'filled');
scatter(C(2, 1), C(2, 2), 'bo', 'filled');
hold off;