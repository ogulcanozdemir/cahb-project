function [modeledData] = prepareFisherVectors(dataPath, models, repeatCount, ...
                                        dimensionRange, dataRange, nSigns, varargin)
% Prepares Fisher Vectors via vl_feat with given GMM means, covariances and priors (model)
% dataPath          : data set path 
% models            : GMM result models 
% repeatCount       : process repeat count (default 5)
% dimensionRange    : one of the predefined dimension range (default = all 41:436)
% dataRange         : range of the data folder such as train or test
% nSigns            : number of signs will be processed

saveName = [];
for i = 1:2:length(varargin),
    name = varargin{i};
    value = varargin{i+1};
    switch name
        case 'save'
            saveFlag = true;
            saveName = value; 
        otherwise
    end
end

if isempty(saveName), saveFlag = false; end
if nargin < 5, nSigns = 2000; end
if nargin < 4, dimensionRange = 41:436; end
if nargin < 3, repeatCount = 5; end
if nargin < 2, disp('models cannot be empty'); return; end
if nargin < 1, disp('dataPath and models cannot be empty'); return; end

tic;
userFolders = dir(fullfile(dataPath)); 
userFolders = userFolders(3:end);
userFolders = userFolders(dataRange);

modeledData = [];
for userIdx = 1:numel(userFolders),
    userDir = fullfile(dataPath, userFolders(userIdx).name);
    signFolders = dir(fullfile(userDir)); 
    signFolders = signFolders(3:end-1);

    for signIdx = 1:nSigns,
        fprintf('Subject %d, Sign %d :', userIdx, signIdx); tic;
        sign = signFolders(signIdx);
        filePath = [userDir filesep sign.name];
        file = dir(fullfile(filePath, ['*' num2str(signIdx) '.mat']));
        featureFile = [filePath filesep file.name];
        load(featureFile);
        
        tmpSample = feature(:, dimensionRange);
        nTraj = size(tmpSample,1);

        for repeatIdx=1:repeatCount,
           model = models(1, repeatIdx); model = model{1,1};
           tmpDataPca = ((tmpSample - repmat(model.M, nTraj, 1)) * model.V);
           fisher_data = vl_fisher(tmpDataPca', model.means, model.covariances, model.priors, 'improved');
           modeledData{1, repeatIdx}{userIdx, signIdx} = fisher_data;
        end
        toc;
    end
end
toc;

if saveFlag,
   save(saveName, 'modeledData', '-v7.3');
end
      
end