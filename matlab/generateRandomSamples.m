function [randomSamples] = generateRandomSamples(featurePath, featureNames, ...
    trajectoryCounts, sampleSize)
    featureSize = 436;
    randomSamples = zeros(sampleSize,featureSize);
    
    numTrajectories = sum(trajectoryCounts);
    sampleRange = sort(randsample(numTrajectories,sampleSize));
    
    baseIdx = 0;
    ceilIdx = 0;
    sampleIdx = 1;
    for i=1:size(featureNames,1)
        ceilIdx = ceilIdx + trajectoryCounts(i);
        featureZippedFilePath = strjoin([featurePath filesep featureNames(i) '.features.gz'],'');      
        gunzip(featureZippedFilePath);
        featureFilePath = featureZippedFilePath(1:end-3);
        localFeatures = dlmread(featureFilePath);
        while sampleIdx <= size(sampleRange,1) && sampleRange(sampleIdx) <= ceilIdx
            localIdx = sampleRange(sampleIdx) - baseIdx;
            randomSamples(sampleRange(sampleIdx),:) = localFeatures(localIdx,:);
            sampleIdx = sampleIdx + 1;
        end
        delete(featureFilePath);
        baseIdx = ceilIdx;
    end

end