from os.path import dirname, abspath, join, sep, realpath
from os import rename
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
    classifier = 'elm'

    exp_id = classifier + "_" + featureName
    logFileName = exp_id + ".log"
    if isLogging:
        old_stdout = sys.stdout
        log_file = open(logFileName, 'w')
        sys.stdout = log_file

    # initialize paths for reading process
    featureAnnotationsDir = join(dirname(realpath(__file__)), 'feature-annotations')

    # read feature annotations of charades dataset
    trainAnnotationsFile = featureAnnotationsDir + sep + 'charades_v04_train.csv'
    testAnnotationsFile = featureAnnotationsDir + sep + 'charades_v04_test.csv'
    trainAnnotations = Utility.readAnnotations(trainAnnotationsFile)
    testAnnotations = Utility.readAnnotations(testAnnotationsFile)

    # read baseline features
    baselineFeaturesDir = join(dirname(realpath(__file__)), 'baseline-features')
    featureFile = baselineFeaturesDir + sep + featureName + '.mat'

    trainData, trainLabels, trainAnnotations_subsampled, testData, testLabels, testAnnotations_subsampled = Utility.readBaselineIDTFeatures(
        featureFile, trainAnnotations, testAnnotations)

    if classifier == 'rf':  # train random forest classifier
        resultsProba, resultsLabels, params = Classifier.trainRandomForest(trainData, trainLabels, testData)
    elif classifier == 'svc':  # train one-vs-rest linear svc with 5-fold cross validation
        resultsProba, resultsLabels, params = Classifier.trainLinearSVC(trainData, trainLabels, testData)
    elif classifier == 'elm':  # train ELM Classifier
        resultsProba, resultsLabels, params = Classifier.trainELMClassifier(trainData, trainLabels, testData)

    parameters = '_'.join('{}_{}'.format(key, val) for key, val in params.items())
    new_exp_id = exp_id + '_' + parameters

    list1 = np.array(testAnnotations_subsampled, dtype=np.object)
    scipy.io.savemat(new_exp_id + '.mat', {'resultsProba': resultsProba, 'resultsLabels': resultsLabels,
                                           'testAnnotations': np.transpose(list1)})

    if isLogging:
        sys.stdout = old_stdout
        log_file.close()

    newLogFileName = new_exp_id + ".log"
    rename(logFileName, newLogFileName)
