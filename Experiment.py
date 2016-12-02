from os.path import dirname, abspath, join, sep
from Utility import Utility
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import sys
from time import time
import scipy.io
import numpy as np

def trainRandomForest(trainData, trainLabels, testData, testLabels):
    print("\nTraining Random Forest Classifier...")

    trainData = np.asarray(trainData)
    trainLabels = np.asarray(trainLabels)

    forest = RandomForestClassifier(n_estimators=1000, random_state=0, verbose=1, n_jobs=-1)
    multi_target_forest = OneVsRestClassifier(forest, n_jobs=-1)
    print("\nClassifier")
    print("----------------------------------")
    print(multi_target_forest)

    t0 = time()
    multi_target_forest.fit(trainData, trainLabels)
    print("\nTraining finished in %0.3fs \n" % (time() - t0))

    t0 = time()
    predictedLabels = multi_target_forest.predict(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))

    print("\nPredicted Labels")
    print("----------------------------------")
    print(predictedLabels)

    t0 = time()
    predictedProba = multi_target_forest.predict_proba(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))

    print("\nPredicted Probabilities")
    print("----------------------------------")
    print(predictedProba)

    return predictedProba, predictedLabels

if __name__ == '__main__':
    isLogging = True

    featureName = "FV_d3_k64" #str(sys.argv[1])

    if isLogging:
        old_stdout = sys.stdout
        log_file = open("Results_" + featureName + ".log", 'w')
        sys.stdout = log_file

    # initialize paths for reading process
    absolutePath = dirname(dirname(dirname(abspath("__file__"))))
    featureAnnotationsDir = join(absolutePath, 'feature-annotations')
    
    # read feature annotations of charades dataset
    trainAnnotationsFile = featureAnnotationsDir + sep + 'charades_v04_train.csv'
    testAnnotationsFile = featureAnnotationsDir + sep + 'charades_v04_test.csv'
    
    trainAnnotations = Utility.readAnnotations(trainAnnotationsFile)
    testAnnotations = Utility.readAnnotations(testAnnotationsFile)
    
    #Utility.saveAnnotations(trainAnnotations, 'trainLabels.csv')
    #Utility.saveAnnotations(testAnnotations, 'testLabels.csv')
    del featureAnnotationsDir, trainAnnotationsFile, testAnnotationsFile
    
    # read baseline features
    baselineFeaturesDir = join(absolutePath, 'baseline-features')
    featureFile = baselineFeaturesDir + sep + 'FV_d3_k64.mat'
    
    trainData, trainLabels, testData, testLabels = Utility.readBaselineIDTFeatures(featureFile, trainAnnotations, testAnnotations)
    del baselineFeaturesDir, featureFile, trainAnnotations, testAnnotations, absolutePath, sep

    resultsProba, resultsLabels = trainRandomForest(trainData, trainLabels, testData, testLabels)

    scipy.io.savemat('Results_FV_d3_k64.mat', {'resultsProba': resultsProba, 'resultsLabels': resultsLabels})

    if isLogging:
        sys.stdout = old_stdout
        log_file.close()
