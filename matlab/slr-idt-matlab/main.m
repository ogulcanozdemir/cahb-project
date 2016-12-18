clear, close all, clc

% main script
% vlfeatPath = ['D:\tools\vlfeat-0.9.20\toolbox' filesep];
idtPathPrefix = 'D:';
idtPath = [idtPathPrefix filesep 'database\experiment_data\slr-idt-data'];
dataPath = ['D:\database\DEVISIGN_L' filesep];

addpath('util');
% run([vlfeatPath 'vl_setup']);

% parameters
fisherDataPath = [idtPath filesep 'fisher'];
svmDataPath = [idtPath filesep 'svm_data_1000_500s'];
nSigns = 1000;
clusterCount = 64;
repeatCount = 5;

%% Prepare component map
cmpKeys = {'hog' }; %, 'hof'  , 'mbh'  , 'hog_hof', 'hog_mbh'       , 'hof_mbh', 'all'};
cmpVals = {41:136}; %, 137:244, 245:436, 41:244   , [41:136 245:436], 137:436  , 41:436};
componentMap = containers.Map(cmpKeys, cmpVals);
                
%% Prepare Leave-One-Out Cross-Validation parameter map
loocvKeys = {'LU1O', 'LU2O', 'LU3O', 'LU4O', 'LU5O', 'LU6O', 'LU7O', 'LU8O'};
loocvTrain = {3:12, [1:2 5:12], [1:4 7:12], [1:6 9:12], [1:8 10:12], [1:9 11:12], [1:10 12], 1:11};
loocvTest  = {1:2 , 3:4       , 5:6       , 7:8       , 9          , 10         , 11       , 12  };
for i=1:numel(loocvKeys), 
    loocvVals{i} = struct('train', loocvTrain{i}, 'test', loocvTest{i});
end
loocvMap = containers.Map(loocvKeys, loocvVals);

%% Extract feature mat files from extracted features for each user and sign
extractIDTFeatures(dataPath, 'processType', 'decompress', 'indexRange', 1:nSigns, 'verbose', true, 'saveFlag', true);

tic;
%% Start to prepare fisher vectors for each fold
for loocvIdx = 1:numel(loocvKeys),
    loocvName = loocvKeys{loocvIdx};
    trainRange = loocvMap(loocvName).train;
    testRange = loocvMap(loocvName).test;
    
    disp(['Preparing data for ' loocvName ' : ']); tic;
    % map extracted features
    [trajectoryMap, numberOfTrajectories] = mapTrajectories(dataPath, trainRange);
    
    % generate random samples from trajectory map
    [randomSamples] = generateRandomSamples(dataPath, trajectoryMap, numberOfTrajectories, clusterCount, repeatCount, nSigns, trainRange);
    clear trajectoryMap numberOfTrajectories;
    
    % Prepare fisher vectors for each component
    for cmpIdx = 1:numel(cmpKeys),
        cmpName = cmpKeys{cmpIdx};
        cmpDim = componentMap(cmpName);
        
        disp(['Preparing data for ' loocvName ' and ' cmpName ' : ']); tic;
        trainName = ['train_' num2str(clusterCount) 'k_' num2str(repeatCount) 'r_' loocvName '_' cmpName];
        testName = ['test_' num2str(clusterCount) 'k_' num2str(repeatCount) 'r_' loocvName '_' cmpName];
        
        % generate models from randomly sampled data
        [models] = generateGMMs(randomSamples, clusterCount, repeatCount, cmpDim, 'pca', true);
        
        % prepare train and test fisher vectors for each sign from original samples
        prepareFisherVectors(dataPath, models, repeatCount, cmpDim, trainRange, nSigns, 'save', [fisherDataPath filesep trainName]);
        prepareFisherVectors(dataPath, models, repeatCount, cmpDim, testRange, nSigns, 'save', [fisherDataPath filesep testName]);
        
        prepareDataForSVM([fisherDataPath filesep trainName], repeatCount, svmDataPath, trainName);
        prepareDataForSVM([fisherDataPath filesep testName], repeatCount, svmDataPath, testName);
        clear modeledData models;
        
        delete([fisherDataPath filesep trainName]);
        delete([fisherDataPath filesep testName]);
    end
    clear randomSamples;
    toc;
end
toc;