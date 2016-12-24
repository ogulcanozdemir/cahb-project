function [fisherVectors] = prepareFisherVectors(featurePath, videoNames, ...
    cmpDim, means, covariances, priors, M, V)

    clusterCount = size(priors,1);
    fisherVectorSize = 2*clusterCount*size(V,2);
    fisherVectors = zeros(length(videoNames),fisherVectorSize);
    
    for i=1:length(videoNames)
        fprintf('idx = %d/%d, videoName = %s\n', i, length(videoNames), videoNames{i});
        featureZippedFilePath = [featurePath filesep videoNames{i} '.features.gz'];      
        gunzip(featureZippedFilePath);
        featureFilePath = featureZippedFilePath(1:end-3);
        localFeatures = dlmread(featureFilePath);
        
        subLocalFeatures = localFeatures(:,cmpDim);
        reducedFeatures = (subLocalFeatures - repmat(M, size(subLocalFeatures, 1), 1)) * V;
        fisherVectors(i,:) = vl_fisher(reducedFeatures', means, covariances, priors)';

        delete(featureFilePath);
    end
    
    
                                        
end