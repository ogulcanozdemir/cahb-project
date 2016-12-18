clear 
close all  
clc

featurePath = [pwd filesep 'charades-features'];
[featureNames, trajectoryCounts] = calculateTrajectoryCounts(featurePath);

%%

sampleSize = 1000;
testSize = 1;
i=1;
% for i = 1:testSize:(size(featureNames,1)-testSize)
testRange = i:i+testSize-1;
testFeatureNames = featureNames(testRange);
testTrajectoryCounts = trajectoryCounts(testRange);
trainingRange = [1:i-1 (i+testSize):size(featureNames,1)];
trainingFeatureNames = featureNames(trainingRange);
trainingTrajectoryCounts = trajectoryCounts(trainingRange);

randomSamples = generateRandomSamples(featurePath, trainingFeatureNames, ...
trainingTrajectoryCounts, sampleSize);
randomSamples;

%%

cmpKeys = {'hog' }; %, 'hof'  , 'mbh'  , 'hog_hof', 'hog_mbh'       , 'hof_mbh', 'all'};
cmpVals = {41:136}; %, 137:244, 245:436, 41:244   , [41:136 245:436], 137:436  , 41:436};
componentMap = containers.Map(cmpKeys, cmpVals);

clusterCount = 64;
repeatCount = 5;

run('vlfeat/toolbox/vl_setup');

for cmpIdx = 1:numel(cmpKeys)
    cmpName = cmpKeys{cmpIdx};
    cmpDim = componentMap(cmpName);
    [models] = generateGMMs(randomSamples, clusterCount, cmpDim);

%     % prepare train and test fisher vectors for each sign from original samples
%     prepareFisherVectors(dataPath, models, repeatCount, cmpDim, trainRange, nSigns, 'save', [fisherDataPath filesep trainName]);
%     prepareFisherVectors(dataPath, models, repeatCount, cmpDim, testRange, nSigns, 'save', [fisherDataPath filesep testName]);
% 
%     prepareDataForSVM([fisherDataPath filesep trainName], repeatCount, svmDataPath, trainName);
%     prepareDataForSVM([fisherDataPath filesep testName], repeatCount, svmDataPath, testName);
%     clear modeledData models;
% 
%     delete([fisherDataPath filesep trainName]);
%     delete([fisherDataPath filesep testName]);
end
    
% end
% toc;