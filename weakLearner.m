function learner = weakLearner(train,label, weights, type)
% INPUTS:
%       train: d by N input data
%       label: N by 1 true labels
%       weights: 1 by N vector contains the weight for each point
%       type: type of the classifier: 'decision_stump' or 'perceptron'
% OUTPUT:
%       learner: model contains the parameters of the weak classifier
% This function is called in AdaBoost function
%========================
% last update: 11/25/2011
%========================
if nargin==4
    switch type
        case 'decision_stump'
            % Check that weights sum to one
            weights = weights/sum(weights);
            % [d ~] = size(train);
            d = size(train, 1);
            % preallocate the parameters
            A = zeros(d,1);
            B = zeros(d,1);
            thr = zeros(d,1);
            er = zeros(d,1);
            % go over all the features and find the parameters
            for i= 1:d
                [A(i), B(i), thr(i), er(i)] = decision_stump(train(i,:),label,weights);
            end
            % [~, bestInd] = min(er);
            [temp_mini bestInd] = min(er);
            thr = thr(bestInd);
            A = A(bestInd);
            B = B(bestInd);
            learner.type = 'decision_stump';
            learner.thr = thr;
            learner.A = A;
            learner.B = B;
            learner.ind = bestInd;
        case 'perceptron'
            % Check that weights sum to one
            weights = weights/sum(weights);
            [w b] = perceptron(train,label,weights);
            learner.type = 'perceptron';
            learner.w = w;
            learner.b = b;
        otherwise
            error('Mode is not supported!');
    end
end
end
%% Decision stump function
function [A,B,thr,error] = decision_stump(tr,label,weight)
% it minimizes sum(weight * |label - (A*(tr>thr) + B)|^2) and returns the
% best values for A,B,thr.
% the prediction based on decision stump is computed: sign(A*(tr>thr) + B)
% mseyed: avoid any for loops in this function since it is called in a for
% loop itself. There is some efficient code to implement this.

% sort the data to find the threshold
[tr index] = sort(tr);
label = label(index);
weight = weight(index);
slw = cumsum(label'.*weight); % sum_label_weight
sw = cumsum(weight); % sum_weight

% compute B
B = slw ./ sw;
sw(end) = 0; % avoid ambiguity at next line
A = (slw(end) - slw) ./ (1-sw) - B;
sw(end) = 1; % set back the original value

% this is the smart way that people use to compute the error efficiently
error = sum(weight.*label'.^2) - 2*A.*(slw(end)-slw) - ...
    2*B*slw(end) + (A.^2 +2*A.*B) .* (1-sw) + B.^2;

% find the best index
[error, k] = min(error);

if k == numel(tr)
    thr = tr(k);
else
    thr = (tr(k) + tr(k+1))/2;
end
A = A(k);
B = B(k);
end
%% Perceptron function (It is using weights)
function [w b] = perceptron(tr , label, weight)

[d N] = size(tr);
% initialize weights and bias
w = zeros(d,1);
b = 0;
maxiter = 3;
for iter = 1 : maxiter
    for n = 1 : N
        % Check for the wrong decision
        if( (w'*tr(:,n) + b) * label(n) <= 0 )
            w = w + label(n) * weight(n) * tr(:,n);
            b = b + weight(n) * label(n);
        end
    end
end
end