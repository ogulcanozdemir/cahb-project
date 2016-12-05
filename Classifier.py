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
    print(trainData.shape)
    print(trainLabels.shape)

    Cs = np.power(2, np.linspace(-3, 9, num=7))

    osvc = OneVsRestClassifier(LinearSVC(class_weight='balanced', verbose=False, multi_class='ovr', max_iter=2000), n_jobs=-1)

    parameters = {
        "estimator__C" : Cs,
    }

    svc = GridSearchCV(osvc, cv=5, param_grid=parameters, n_jobs=-1)

    t0 = time()
    svc.fit(trainData, trainLabels)
    print("\nTraining finished in %0.3fs \n" % (time() - t0))

    print("Best parameters: ")
    print(svc.best_params_)
    print("\nBest estimator: ")
    print(svc.best_estimator_)
    print("Best score: ")
    print(svc.best_score_)

    t0 = time()
    confidence_scores = svc.decision_function(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))

    t0 = time()
    predictedLabels = svc.predict(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))


    print("\nPredicted Labels")
    print("----------------------------------")
    print(predictedLabels)

    print("\nConfidence Scores")
    print("----------------------------------")
    print(confidence_scores)

    return confidence_scores, predictedLabels