function [fisherVectors] = prepareFisherVectors(featurePath, videoNames, ...
    cmpDim, means, covariances, priors, M, V)

    clusterCount = size(priors,1);
    fisherVectorSize = 2*clusterCount*size(V,2);
    fisherVectors = zeros(length(videoNames),fisherVectorSize);
    
    for i=1:length(videoNames)
        featureZippedFilePath = strjoin([featurePath filesep videoNames(i) '.features.gz'],'');      
        gunzip(featureZippedFilePath);
        featureFilePath = featureZippedFilePath(1:end-3);
        localFeatures = dlmread(featureFilePath);
        
        reducedFeatures = (localFeatures(:,cmpDim)-M)*V;
        fisherVectors(i,:) = vl_fisher(reducedFeatures', means, covariances, priors)';

        delete(featureFilePath);
    end
    
    
                                        
end