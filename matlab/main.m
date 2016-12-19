clear 
close all  
clc
tic;

featurePath = [pwd filesep 'charades-features'];
[videoNames, trajectoryCounts] = calculateTrajectoryCounts(featurePath);
toc;
%%

sampleSize = pow2(15);
testSize = 1;
i= randsample(1:testSize:(size(videoNames,1)-testSize),1);
% for i = 1:testSize:(size(featureNames,1)-testSize)
testRange = i:i+testSize-1;
testVideoNames = videoNames(testRange);
testTrajectoryCounts = trajectoryCounts(testRange);
trainingRange = [1:i-1 (i+testSize):size(videoNames,1)];
trainingVideoNames = videoNames(trainingRange);
trainingTrajectoryCounts = trajectoryCounts(trainingRange);

randomSamples = generateRandomSamples(featurePath, trainingVideoNames, ...
trainingTrajectoryCounts, sampleSize);
toc;

%%

cmpDim = 41:436;
clusterCount = 64;
repeatCount = 5;

run('vlfeat/toolbox/vl_setup');

[V, ~, M] = pca2(randomSamples(:,cmpDim), 0.99);
reducedSamples = (randomSamples(:,cmpDim)-M)*V;
[means, covariances, priors] = vl_gmm(reducedSamples', ...
    clusterCount, 'MaxNumIterations', 10000);
toc;
%%
fisherVectors = prepareFisherVectors(featurePath, videoNames, cmpDim, ...
    means, covariances, priors, M, V);
toc;
%%
testFisherVectors = fisherVectors(testRange,:);
trainingFisherVectors = fisherVectors(trainingRange,:);

model = KDTreeSearcher(trainingFisherVectors);
[n,d] = knnsearch(model,testFisherVectors);
videoNames(testRange)
videoNames(n)
toc;