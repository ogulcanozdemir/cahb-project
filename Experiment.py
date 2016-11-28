from os.path import dirname, abspath, join, sep
from Utility import Utility
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

def trainRandomForest(trainData, trainLabels, testData, testLabels):
    forest = RandomForestClassifier(n_estimators=100, random_state=1, verbose=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    multi_target_forest.fit(trainData, trainLabels)
    results = multi_target_forest.predict(testData)
    
    return results

if __name__ == '__main__':
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
    
    trainData, trainLabels, testData, testLabels = Utility.readBaselineIDTFeatures(featureFile,
                                                                           trainAnnotations,
                                                                           testAnnotations)
    del baselineFeaturesDir, featureFile, trainAnnotations, testAnnotations, absolutePath, sep
    
    
    # train
    #from sklearn.multiclass import OneVsRestClassifier
    #from sklearn.svm import SVC
    #rc = OneVsRestClassifier(SVC(kernel='linear'))
    #rc.fit(trainData, trainLabels)
    #rc.predict(testData)
    
    ################################
    results = trainRandomForest(trainData, trainLabels, testData, testLabels)
