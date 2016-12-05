from os.path import dirname, abspath, join, sep
from Utility import Utility
import Classifier
import sys
import scipy.io
import warnings
import numpy as np

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    isLogging = True

    featureName = "FV_d3_k128"

    if isLogging:
        old_stdout = sys.stdout
        log_file = open("Results_" + featureName + "_elm.log", 'w')
        sys.stdout = log_file

    # initialize paths for reading process
    absolutePath = dirname(dirname(dirname(abspath("__file__"))))
    featureAnnotationsDir = join(absolutePath, 'feature-annotations')
    
    # read feature annotations of charades dataset
    trainAnnotationsFile = featureAnnotationsDir + sep + 'charades_v04_train.csv'
    testAnnotationsFile = featureAnnotationsDir + sep + 'charades_v04_test.csv'
    trainAnnotations = Utility.readAnnotations(trainAnnotationsFile)
    testAnnotations = Utility.readAnnotations(testAnnotationsFile)

    # read baseline features
    baselineFeaturesDir = join(absolutePath, 'baseline-features')
    featureFile = baselineFeaturesDir + sep + featureName + '.mat'

    trainData, trainLabels, trainAnnotations_subsampled, testData, testLabels, testAnnotations_subsampled = Utility.readBaselineIDTFeatures(featureFile, trainAnnotations, testAnnotations)

    # train random forest classifier
    #resultsProba, resultsLabels = Classifier.trainRandomForest(trainData, trainLabels, testData, testLabels)
    #scipy.io.savemat('rf_' + featureName + '.mat', {'resultsProba': resultsProba, 'resultsLabels': resultsLabels})

    # train one-vs-rest linear svc with 5-fold cross validation
    #resultsProba, resultsLabels = Classifier.trainLinearSVC(trainData, trainLabels, testData, testLabels)

    # train ELM Classifier
    resultsProba, resultsLabels = Classifier.trainELMClassifier(trainData, trainLabels, testData, testLabels)

    list1 = np.array(testAnnotations_subsampled, dtype=np.object)
    scipy.io.savemat('elm_' + featureName + '.mat', {'resultsProba': resultsProba, 'resultsLabels': resultsLabels, 'testAnnotations': list1.T})

    if isLogging:
        sys.stdout = old_stdout
        log_file.close()
