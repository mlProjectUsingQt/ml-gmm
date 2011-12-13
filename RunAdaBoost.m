function predictions = RunAdaBoost(data,model)
% INPUTS:
%       data: d by N input data
%       model: the model which is learned in AdaBoost function
% OUTPUT:
%       predictions: predicted labels for input data
%========================
% last update: 11/25/2011
%========================
strongClassifier = zeros(size(data,2),1);
switch lower(model.type)
    case 'decision_stump'
        for i = 1:length(model.error)
            weakClassifier = sign(model.learner{i}.A *...
                (data(model.learner{i}.ind,:)> model.learner{i}.thr)+ ...
                model.learner{i}.B);
            strongClassifier = strongClassifier + ...
                model.alpha{i}* weakClassifier';
        end
    case 'perceptron'
        for i = 1:length(model.error)
            weakClassifier = sign( model.learner{i}.w' * ...
                data + model.learner{i}.b);
            strongClassifier = strongClassifier + ...
                model.alpha{i}* weakClassifier';
        end
    otherwise
        error ('mode is not supported')
end
predictions = sign(strongClassifier);