clear all, close all, clc

videoPath = [pwd filesep 'charades-data'];
featurePath = [pwd filesep 'charades-features'];
executableName = 'DenseTrackStab';

videoDir = dir(videoPath); 
videoDir = videoDir(3:end);
featureDir = dir(featurePath); 
featureDir = featureDir(3:end);

parfor videoIdx=1:numel(videoDir),
    videoFileName = videoDir(videoIdx).name;
    videoFilePath = [videoPath filesep videoFileName];
    featureFilePath = [featurePath filesep videoFileName(1:end-4) '.features.gz'];
    disp(['sudo ./' executableName ' ' videoFilePath ' | gzip > ' featureFilePath])
end