function [features] = generateRandomSamples(dataPath, trajectoryMap, ... 
                            numberOfTrajectories, clusterCount, repeatCount, nSigns, ...
                            dataRange, varargin)
% Generates random uniform samples from given data
% dataPath              : data set path
% trajectoryMap         : map created created with mapTrajectories function
% numberOfTrajectories  : trajectory counts created with mapTrajectories function
% clusterCount          : cluster count defined for trajectory size (64,
%                         128, 256, 512 etc.) (default = 64)
% repeatCount           : predefined number of sampling calculations (default = 5)
% nSigns                : number of signs (default = 200)
% dataRange             : range of users
% saveFlag              : saves random samples (default = false)

saveName = [];
saveFlag = [];
for i = 1:2:length(varargin),
    name = varargin{i};
    value = varargin{i+1};
    switch name
        case 'save'
            saveName = value; 
            saveFlag = true;
        otherwise
    end
end

if isempty(saveName), saveFlag = false; end
if nargin < 7, dataRange = 1:8; end
if nargin < 6, nSigns = 200; end
if nargin < 5, repeatCount = 5; end
if nargin < 4, clusterCount = 64; end
if nargin < 3, disp('numberOfTrajectories cannot be empty, see mapTrajectories function.'); return; end
if nargin < 2, disp('trajectoryMap cannot be empty, see mapTrajectories function.'); return; end
if nargin < 1, disp('dataPath cannot be empty.'); return; end

disp('Started generating random samples...')
tic;
sampleCoeff = 1000;
featureDimension = 436; % predefined features dimension (IDTs are 436 dimensional)

totalTrajectories = size(trajectoryMap, 1);

% Uniformly sample random indices
randomSampleIndex = [];
for repeatIdx = 1:repeatCount,
   rng('shuffle', 'v5uniform');
   randomSampleIndex = [randomSampleIndex, randi(totalTrajectories, 1, clusterCount * sampleCoeff)];
end
samples = zeros(clusterCount * repeatCount * sampleCoeff, featureDimension);

userFolders = dir(fullfile(dataPath)); 
userFolders = userFolders(3:end);
userFolders = userFolders(dataRange);
for userIdx = 1:numel(userFolders),
    userDir = fullfile(dataPath, userFolders(userIdx).name);
    signFolders = dir(fullfile(userDir)); 
    signFolders = signFolders(3:end-1);
    
    for signIdx = 1:nSigns,
        sign = signFolders(signIdx);
        filePath = [userDir filesep sign.name];
        file = dir(fullfile(filePath, ['*' num2str(signIdx) '.mat']));
        featureFile = [filePath filesep file.name];
        load(featureFile);
        
        idx = find(numberOfTrajectories(:, 1) == dataRange(userIdx) & numberOfTrajectories(:, 2) == signIdx);
        if idx == 1, tRangeLow = 0; else tRangeLow = numberOfTrajectories(idx-1, 3); end
        tRangeHigh = numberOfTrajectories(idx, 3);
        
        samplesIdx = find(randomSampleIndex >= tRangeLow + 1 & randomSampleIndex <= tRangeHigh);
        samples(samplesIdx, :) = feature(trajectoryMap(randomSampleIndex(samplesIdx), 3), :);
    end
end

lowerRange = 1;
for repeatIdx=1:repeatCount,
    upperRange = clusterCount * repeatIdx * sampleCoeff;
    range = lowerRange:upperRange;
    features{repeatIdx} = samples(range, :);
    lowerRange = upperRange + 1;
end
toc;

if saveFlag == true,
   save([pwd filesep 'temp' filesep 'random_samples' filesep ...
                    saveName, '_', num2str(repeatCount), 'r.mat'], ...
                    'features', '-v7.3'); 
end

end