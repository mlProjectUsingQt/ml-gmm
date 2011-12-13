

% %% load data
% % mseyed : Still working on this dataset. May be remove it later %==============
% % training data
% data = load('segment.dat');% http://archive.ics.uci.edu/ml/datasets/Statlog+%28Image+Segmentation%29
% trX = data(1:2000,1:end-1)';
% trY = 2*double(data(1:2000,end)<4)-1;
% % testing data
% teX = data(2001:end,1:end-1)';
% teY = 2*double(data(2001:end,end)<4)-1;
% %===========================

%% load data
% This is the texture dataset. mseyed: provide the images later.
fprintf ('Loading data...');
data = load('star.mat');
trX = data.X;
trY = data.D';
data = load('star_test.mat');
teX = data.X;
teY = data.D';
fprintf('done\n');
% %% train classifier
% % uniform distribution
% d = size(trX);
% distribution =  ones(1,d(2))/d(2);
% % training
% tic;
% model = AdaBoost(trX,trY,distribution,30,'decision_stump');
% train_time_Original = toc;
% % Performance for training data
% trP_Original = RunAdaBoost(trX,model);
% trPerformance_Original = mean(trP_Original==trY);
% % Performance for testing data
% teP_Original = RunAdaBoost(teX,model);
% tePerformance_Original = mean(teP_Original==teY);

%% Subsampling using perceptron

% % Run perceptron on original dataset
% [w b] = Perceptron(trX , trY);
% % Compute distance of each point from the hyperplane
% DistanceToHyperplane = abs (w'*trX + b);
% % Find the points that are close to the threshold
% ClosePoints = (DistanceToHyperplane <= 20);
% % Select the close points
% SubsampledSet = trX(:, ClosePoints);
% SubsampledLabel = trY(ClosePoints);
% 
% % uniform distribution
% d = size(SubsampledSet);
% distribution =  ones(1,d(2))/d(2);

trX = trX(:, 1: 100);
trY = trY(1: 100);
K = 10;
gmm_total_start_time = tic;
[SubsampledSet Z] = gmm(trX', K);
gmm_total_time = toc(gmm_total_start_time);
fprintf('GMM total time (s): %f\n', gmm_total_time);
gmm_weight_start_time = tic;
[SubsampledLabel distribution I] = gmm_weight(K, Z, trY);
gmm_weight_time = toc(gmm_weight_start_time);
fprintf('GMM weight time (s): %f\n', gmm_weight_time);
SubsampledSet = SubsampledSet';

% Run the AdaBoost on subsampled dataset
tic;
model = AdaBoost(SubsampledSet,SubsampledLabel,distribution,30,'decision_stump');
train_time_subsample = toc;

% Performance for training data (original set)
trP_subsample = RunAdaBoost(trX,model);
trPerformance_subsample = mean(trP_subsample==trY);

% Performance for testing data (original set)
teP_subsample = RunAdaBoost(teX,model);
tePerformance_subsample = mean(teP_subsample==teY);

%% Comparison
% fprintf('==============================\n');
% fprintf('*** Using original data for training ***\n'); 
% fprintf('Training time (s): %f\n' , train_time_Original);
% fprintf('Performance on training set: %f \n', trPerformance_Original);
% fprintf('Performance on testing set: %f \n', tePerformance_Original);
fprintf('==============================\n');
fprintf('*** Using subsampled data for training ***\n'); 
fprintf('Training time (s): %f\n' , train_time_subsample);
fprintf('Performance on training set: %f \n', trPerformance_subsample );
fprintf('Performance on testing set: %f \n', tePerformance_subsample);

%% Display output images (testing)

numPixels = 256*256;
% input images
figure(1);
for i = 1:8
    subplot(4,2,i);
    imagesc(reshape(teX(21,(i-1)*numPixels+1:i*numPixels),256,256));
    colormap gray;
    axis off;
end

% output images (first scenario)
figure(2);
for i = 1:8
    subplot(4,2,i);
    imagesc(reshape(teP_Original((i-1)*numPixels+1:i*numPixels),256,256));
    colormap gray;
    axis off;
end
    
% output images (second scenario)
figure(3);
for i = 1:8
    subplot(4,2,i);
    imagesc(reshape(teP_subsample((i-1)*numPixels+1:i*numPixels),256,256));
    colormap gray;
    axis off;
end