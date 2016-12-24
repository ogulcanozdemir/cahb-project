function [featureNames, trajectoryCounts] = calculateTrajectoryCounts(featurePath, unzipFlag)

featureDir = dir(fullfile(featurePath, '*.gz'));

featureNames = cell(numel(featureDir),1);
trajectoryCounts = zeros(numel(featureDir),1);

for featureIdx=1:numel(featureDir)
    featureZippedFileName = featureDir(featureIdx).name;
    featureZippedFilePath = [featurePath filesep featureZippedFileName];
    
    [~, featureFileName, ~] = fileparts(featureZippedFileName);
    featureName = featureFileName(1:end-9);
    featureNames{featureIdx} = featureName;

    if unzipFlag,
        fprintf('idx = %d/%d, videoName = %s\n', featureIdx, numel(featureDir), featureName);
        gunzip(featureZippedFilePath);
        featureFilePath = [featurePath filesep featureFileName];
        feature = dlmread(featureFilePath);
        trajectoryCounts(featureIdx) = size(feature(:, :), 1);
        delete(featureFilePath);
    end
end

end