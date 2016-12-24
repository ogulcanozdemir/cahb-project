clear 
close all  
clc
tic;

featurePath = [pwd filesep 'charades-features'];
[videoNames, trajectoryCounts] = calculateTrajectoryCounts(featurePath);
save('trajectoryCounts.mat','-v7.3');
toc;
%%
tic;
sampleSize = pow2(20);
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
save('randomSamples.mat','-v7.3');
toc;

%%
tic;
cmpDim = 41:436;
clusterCount = 64;
maxIterations = pow2(15);

run('vlfeat/toolbox/vl_setup');

[V, ~, M] = pca2(randomSamples(:,cmpDim), 0.99);
reducedSamples = (randomSamples(:,cmpDim)-M)*V;
[means, covariances, priors] = vl_gmm(reducedSamples', ...
    clusterCount, 'MaxNumIterations', maxIterations);
save('gmms.mat','-v7.3');
toc;
%%
tic;
fisherVectors = prepareFisherVectors(featurePath, videoNames, cmpDim, ...
    means, covariances, priors, M, V);
save('fishers.mat','-v7.3');
save('fv.mat','videoNames','fisherVectors');
toc;
%%
tic;
testFisherVectors = fisherVectors(testRange,:);
trainingFisherVectors = fisherVectors(trainingRange,:);
model = KDTreeSearcher(trainingFisherVectors);
[n,d] = knnsearch(model,testFisherVectors);
videoNames(testRange)
videoNames(n)
toc;    
