function [featureNames, trajectoryCounts] = calculateTrajectoryCounts(featurePath)

featureDir = dir(featurePath);
featureDir = featureDir(3:end);

featureNames = cell(numel(featureDir),1);
trajectoryCounts = zeros(numel(featureDir),1);
for featureIdx=1:numel(featureDir)
    featureZippedFileName = featureDir(featureIdx).name;
    featureZippedFilePath = [featurePath filesep featureZippedFileName];
    
    gunzip(featureZippedFilePath);
    featureFileName =  featureZippedFileName(1:end-3);
    featureFilePath = [featurePath filesep featureFileName];
    feature = dlmread(featureFilePath);
    
    featureName = featureFileName(1:end-9);
    featureNames{featureIdx} = featureName;
    trajectoryCounts(featureIdx) = size(feature(:, :), 1);
    delete(featureFilePath);
end


end