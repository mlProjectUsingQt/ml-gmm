function model=AdaBoost(train,train_label, distribution, iteration,weaklearner)
% INPUTS:
%       train: d by N input data--> d is the dimension and N is the number
%       of samples
%       train_label: N by 1 label vector
%       iteration: maximum number of iteration for boosting algorithm
%       weaklearner: type of the weak learner. This code supports two
%       types: 'decision_stump' & 'perceptron'
% OUTPUT:
%       model: the boosted model that can be used in RunAdaBoost function
%========================
% last update: 11/25/2011
%========================

    fprintf('running AdaBoost algorithm...\n');
%     d = size(train);
    % initialize weights uniformly
% 	distribution = ones(1,d(2))/d(2);
    model.error = {};
    model.alpha = {};
    model.learner = {};
    model.type = weaklearner;
    
    for j = 1:iteration
        % learn the weak classifier
        learner = weakLearner(train,train_label,distribution,weaklearner);
        if strcmpi(weaklearner,'decision_stump')
            % use the weak classifier to predict the labels
            predicted = sign(learner.A * (train(learner.ind,:)>learner.thr) + learner.B);
            % compute the weighted error
            er = sum(distribution.*(predicted~=train_label'));
            if er>.5
                break;
            end
            model.error{j} = er;
            % compute alpha, the weight of the classifier
            model.alpha{j} = .5*log((1-model.error{j})/model.error{j});
            model.learner{j} = learner;
            % update weights
            distribution = distribution.*exp(-model.alpha{j}*train_label'.*predicted);
            distribution = distribution/sum(distribution);
        elseif strcmpi(weaklearner,'perceptron')
            % use the weak classifier to predict the labels
            predicted = sign(learner.w'*train + learner.b );
            % compute the weighted error
            er = sum(distribution.*(predicted~=train_label'));
            if er>.5
                break;
            end
            model.error{j} = er;
            % compute alpha, the weight of the classifier
            model.alpha{j} = .5*log((1-model.error{j})/model.error{j});
            model.learner{j} = learner;
            % update weights
            distribution = distribution.*exp(-model.alpha{j}*train_label'.*predicted);        
            distribution = distribution/sum(distribution);
        else
            error ('mode is not supported');
        end
        fprintf('iteration No. %d \n' , j);
    end