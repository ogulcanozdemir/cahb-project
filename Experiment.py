from os.path import dirname, abspath, join, sep
from Utility import Utility
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import sys
from time import time
import scipy.io

def trainRandomForest(trainData, trainLabels, testData, testLabels):
    print("\nTraining Random Forest Classifier...")
    forest = RandomForestClassifier(n_estimators=100, random_state=1, verbose=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    print("\nClassifier")
    print("----------------------------------")
    print(multi_target_forest)

    t0 = time()
    multi_target_forest.fit(trainData, trainLabels)
    print("\nTraining finished in %0.3fs \n" % (time() - t0))

    t0 = time()
    predictedLabels = multi_target_forest.predict_proba(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))

    print("\nPredicted Labels")
    print("----------------------------------")
    print(predictedLabels)

    return predictedLabels

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
    
    Utility.saveAnnotations(trainAnnotations, 'trainLabels.csv')
    Utility.saveAnnotations(testAnnotations, 'testLabels.csv')
    del featureAnnotationsDir, trainAnnotationsFile, testAnnotationsFile
    
    # read baseline features
    baselineFeaturesDir = join(absolutePath, 'baseline-features')
    featureFile = baselineFeaturesDir + sep + 'FV_d3_k64.mat'
    
    trainData, trainLabels, testData, testLabels = Utility.readBaselineIDTFeatures(featureFile, trainAnnotations, testAnnotations)
    del baselineFeaturesDir, featureFile, trainAnnotations, testAnnotations, absolutePath, sep

    results = trainRandomForest(trainData, trainLabels, testData, testLabels)

    scipy.io.savemat('Results_FV_d3_k64.mat', {'results': results})

    if isLogging:
        sys.stdout = old_stdout
        log_file.close()
