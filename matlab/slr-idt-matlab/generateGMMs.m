function [models] = generateGMMs(randomSamples, clusterCount, repeatCount, dimensionRange, varargin)
% Generate GMMs from randomly sampled data
% randomSamples     : subsampled data
% clusterCount      : number of clusters
% repeatCount       : process repeat count
% dimensionRange    : predefined range of dimensions (default = 41:436)
% pca               : default = true
% saveFlag          : default = false

pca = [];
saveName = [];
saveFlag = [];
for i = 1:2:length(varargin),
    name = varargin{i};
    value = varargin{i+1};
    switch name
        case 'pca'
            pca = value;
        case 'save'
            saveFlag = true;
            saveName = value;
        otherwise
    end
end

if isempty(pca), pca = true; end
if isempty(saveName), saveFlag = false; end

featureDimension = size(randomSamples{1,1}, 2) - 40;

if nargin < 4, dimensionRange = 41:featureDimension; end;
if nargin < 3, repeatCount = 5; end
if nargin < 2, clusterCount = 64; end
if nargin < 1, disp('randomSamples cannot be empty'); return; end

models = [];
for repeatIdx = 1:repeatCount,
    subsampledData = cell2mat(randomSamples(repeatIdx));
    subsampledData = subsampledData(:, dimensionRange);
    
    if pca == true,
        fprintf(['Started PCA (repeat:' num2str(repeatIdx) ') : ']); tic;
        [V, ~, M] = pca2(subsampledData, 0.99);
        subsampledData = (subsampledData - repmat(M, size(subsampledData, 1), 1)) * V; toc;
    end
    
    fprintf(['Started GMM (repeat:' num2str(repeatIdx) ') : ' ]); tic;
    [means, covariances, priors] = vl_gmm(subsampledData', clusterCount, 'MaxNumIterations', 10000);
    toc;
    
    models{repeatIdx} = struct('means', means, 'covariances', covariances, 'priors', priors, 'V', V, 'M', M);
end

if saveFlag,
    save([pwd filesep 'temp' filesep 'model' filesep ...
            saveName '_' num2str(repeatCount) 'r.mat'], 'models', '-v7.3');
end

end