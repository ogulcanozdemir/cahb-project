function [means, covariances, priors] = generateGMMs(randomSamples, clusterCount, dimensionRange)
    
[V, ~, M] = pca2(randomSamples, 0.99);
[means, covariances, priors] = vl_gmm(randomSamples', clusterCount, 'MaxNumIterations', 10000);
end