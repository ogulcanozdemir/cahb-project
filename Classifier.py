from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from elm import GenELMClassifier
from random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.svm import LinearSVC
from time import time
import numpy as np


def trainRandomForest(trainData, trainLabels, testData):
    print("\nTraining Random Forest Classifier...")

    trainData = np.asarray(trainData)
    trainLabels = np.asarray(trainLabels)

    ne = 1000
    rs = 0

    forest = RandomForestClassifier(n_estimators=ne, random_state=rs, verbose=1, n_jobs=-1)
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

    params = {
        'ne': ne,
        'rs': rs,
    }

    return predictedProba, predictedLabels, params


def trainLinearSVC(trainData, trainLabels, testData):
    print("\nTraining Linear SVC...")

    trainData = np.asarray(trainData)
    trainLabels = np.asarray(trainLabels)
    print(trainData.shape)
    print(trainLabels.shape)

    iter = 2000
    cross_val = 5

    Cs = np.power(2, np.linspace(-3, 9, num=7))
    parameters = {
        "estimator__C" : Cs,
    }

    osvc = OneVsRestClassifier(LinearSVC(class_weight='balanced', verbose=False, multi_class='ovr', max_iter=iter), n_jobs=-1)
    svc = GridSearchCV(osvc, cv=cross_val, param_grid=parameters, n_jobs=-1)

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
    predictedLabels = svc.predict(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))

    t0 = time()
    confidence_scores = svc.decision_function(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))

    print("\nPredicted Labels")
    print("----------------------------------")
    print(predictedLabels)

    print("\nConfidence Scores")
    print("----------------------------------")
    print(confidence_scores)

    params = {
        'iter': iter,
        'cv': cross_val,
    }

    return confidence_scores, predictedLabels, params


def trainELMClassifier(trainData, trainLabels, testData):
    print("\nTraining ELM Classifier...")

    trainData = np.asarray(trainData)
    trainLabels = np.asarray(trainLabels)
    print(trainData.shape)
    print(trainLabels.shape)

    # create initialize elm activation functions
    nh = 100
    activation = 'tanh'

    if activation == 'rbf':
        act_layer = RBFRandomLayer(n_hidden=nh, random_state=0, rbf_width=0.001)
    elif activation == 'tanh':
        act_layer = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
    elif activation == 'tribas':
        act_layer = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
    elif activation == 'hardlim':
        act_layer = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

    # initialize ELM Classifier
    elm = GenELMClassifier(hidden_layer=act_layer)

    t0 = time()
    elm.fit(trainData, trainLabels)
    print("\nTraining finished in %0.3fs \n" % (time() - t0))

    t0 = time()
    predictedLabels = elm.predict(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))

    t0 = time()
    confidence_scores = elm.decision_function(testData)
    print("\nTesting finished in %0.3fs" % (time() - t0))

    print("\nPredicted Labels")
    print("----------------------------------")
    print(predictedLabels)

    print("\nConfidence Scores")
    print("----------------------------------")
    print(confidence_scores)

    params = {
        'nh': nh,
        'af': activation,
    }

    return confidence_scores, predictedLabels, params
