function generateFisherVectors(nCluster, samplingRate, calcTrajCount)

disp(['Generating Fisher Vectors for parameters: k = ', num2str(nCluster), ....
        ', sampling rate = ', num2str(samplingRate)]);
    
%% Prepare paths
addpath('util');
addpath('features');
run([pwd filesep 'util' filesep 'vlfeat' filesep 'toolbox' filesep 'vl_setup']);

[parentDir, ~, ~] = fileparts(fileparts(pwd));
trainingFeaturePath = [parentDir filesep 'training'];
testFeaturePath = [parentDir filesep 'test'];

%% Calculate Trajectory Counts
if calcTrajCount,
    fprintf('Calculating trajectory counts : '); tic;
    [trainVideoNames, trajectoryCounts] = calculateTrajectoryCounts(trainingFeaturePath, true);
    [testVideoNames, ~] = calculateTrajectoryCounts(testFeaturePath, false);
    save(['features' filesep 'trajectoryCounts.mat'], 'trainVideoNames', ...
                    'trajectoryCounts', 'testVideoNames', '-v7.3');
    toc;
else
    load(['features' filesep 'trajectoryCounts.mat']);
end

%% Generate Random Party
sampleSize = nCluster * samplingRate;
randomSamples = generateRandomSamples(trainingFeaturePath, trainVideoNames, ...
    trajectoryCounts, sampleSize);
save(['features' filesep 'randomSamples_k', num2str(nCluster), '_', num2str(samplingRate) '.mat'],...
                'randomSamples', '-v7.3');

%% Prepare component map
cmpKeys = {'hog', 'hof', 'mbh', 'all'};
cmpValues = {41:136, 137:244, 245:436, 41:436};   
componentMap = containers.Map(cmpKeys, cmpValues);

%% Prepare fisher vectors for each component
for cmpIdx = 1:numel(cmpKeys),
    cmpName = cmpKeys{cmpIdx};
    cmpDim = componentMap(cmpName);
 
    % apply pca
    fprintf(['Started PCA (component:' cmpName ', k:', num2str(nCluster), ') : ']); tic;
    subsampledData = randomSamples(:, cmpDim);
    [V, ~, M] = pca2(subsampledData, 0.99);
    reducedSamples = (subsampledData - repmat(M, size(subsampledData, 1), 1)) * V;
    toc;

    % generate models from randomly sampled data
    fprintf(['Started GMM (component:' cmpName ', k:', num2str(nCluster), ') : ' ]); tic;
    [means, covariances, priors] = vl_gmm(reducedSamples', ...
                        nCluster, 'MaxNumIterations', 10000);
    toc;
    
    % prepare train and test fisher vectors
    fprintf(['Started Training Fisher (component:' cmpName ', k:', num2str(nCluster), ') : ' ]); tic;
    trainFisherVectors = prepareFisherVectors(trainingFeaturePath, trainVideoNames, cmpDim, ...
                        means, covariances, priors, M, V); toc;
    save(['features' filesep 'train_fisher_k', num2str(nCluster), '_', cmpName, '_', num2str(samplingRate), '.mat'], ...
                           'trainFisherVectors', 'trainVideoNames', '-v7.3');    
                       
    fprintf(['Started Training Fisher (component:' cmpName ', k:', num2str(nCluster), ') : ' ]); tic;
    testFisherVectors = prepareFisherVectors(testFeaturePath, testVideoNames, cmpDim, ...
                        means, covariances, priors, M, V); toc;
    save(['features' filesep 'test_fisher_k', num2str(nCluster), '_', cmpName, '_', num2str(samplingRate), '.mat'], ...
                           'testFisherVectors', 'testVideoNames', '-v7.3');
end

clear randomSamples;
end

