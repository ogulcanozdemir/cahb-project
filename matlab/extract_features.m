clear all, close all, clc

videoPath = [pwd filesep 'charades-data'];
featurePath = [pwd filesep 'charades-features'];
executableName = 'DenseTrackStab';

videoDir = dir(videoPath); 
videoDir = videoDir(3:end);
featureDir = dir(featurePath); 
featureDir = featureDir(3:end);

nTrajectories = [];
for videoIdx=1:numel(videoDir)
    videoFileName = videoDir(videoIdx).name;
    videoFilePath = [videoPath filesep videoFileName];
    featureZippedFilePath = [featurePath filesep videoFileName(1:end-4) '.features.gz'];
    
    gunzip(featureZippedFilePath);
    featureFilePath = [featurePath filesep videoFileName(1:end-4) '.features'];
    feature = dlmread(featureFilePath);
    
    t = {videoFileName(1:end-4), size(feature(:, :), 1)};
    nTrajectories = [nTrajectories; t];
    delete(featureFilePath);
end
save('nTrajectories.mat', 'nTrajectories', '-v7.3');