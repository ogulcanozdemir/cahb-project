from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from time import time
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

def trainLinearSVC(trainData, trainLabels, testData, testLabels):
    print("\nTraining Linear SVC...")

    trainData = np.asarray(trainData)
    trainLabels = np.asarray(trainLabels)

    Cs = np.power(2, np.linspace(-3, 9, num=7))
    osvc = OneVsRestClassifier(LinearSVC(class_weight='balanced', verbose=False), n_jobs=-1)
    svc = GridSearchCV(osvc, cv=5, param_grid=dict(C=Cs), n_jobs=-1)

    t0 = time()
    svc.fit(trainData, trainLabels.ravel())
    print("\nTraining finished in %0.3fs \n" % (time() - t0))

    t0 = time()
    predictedLabels = svc.predict(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))

    t0 = time()
    predictedProba = svc.predict_proba(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))

    t0 = time()
    predictedLogProba = svc.predict_log_proba(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))

    print("\nPredicted Labels")
    print("----------------------------------")
    print(predictedLabels)

    print("\nPredicted Probabilities")
    print("----------------------------------")
    print(predictedProba)

    return predictedProba, predictedLogProba, predictedLabels