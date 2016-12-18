import numpy as np
import h5py
import sys
import os
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from time import time

if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    isLogging = True

    trainPrefix = 'train_data_'
    testPrefix = 'test_data_'
    resultPrefix = 'result_'

    components = ['hog', 'hof', 'mbh', 'hog_hof', 'hog_mbh', 'hof_mbh', 'all']
    nCluster = 64
    repeatCount = 5
    foldNames = ['LU1O', 'LU2O', 'LU3O', 'LU4O', 'LU5O', 'LU6O', 'LU7O', 'LU8O']

    dataPath = '\\\\DESKTOP-J7AKN5S\\svm_data_2000\\'

    # noinspection PyTypeChecker
    Cs = np.power(2, np.linspace(-3, 9, num=7))

    for foldName in foldNames:

        if not os.path.exists(foldName):
            os.makedirs(foldName)

        for component in components:
            componentSuffix = str(nCluster) + 'k_' + str(repeatCount) + 'r_' + foldName + '_' + str(component) + '_'
            trainDataName = 'train_' + componentSuffix
            testDataName = 'test_' + componentSuffix
            resultName = foldName + '\\' + 'result_' + componentSuffix

            for index in range(1, repeatCount+1):
                if isLogging:
                    old_stdout = sys.stdout
                    log_file = open(resultName + str(index) + ".log", 'w')
                    sys.stdout = log_file

                print("====================================================")
                print("SVM Classification #" + str(index))
                print("====================================================")
                print("====================================================")

                # start training
                t0 = time()
                filename = dataPath + trainDataName + str(index) + '.mat'
                file = h5py.File(filename, 'r')

                # read training data
                trainData = np.transpose(np.array(file.get('data')))
                trainLabels = np.transpose(np.array(file.get('labels')))

                # start grid search to find best regularization parameters C
                svc = GridSearchCV(LinearSVC(class_weight='balanced', verbose=False), cv=5, param_grid=dict(C=Cs), n_jobs=-1)
                svc.fit(trainData, trainLabels.ravel())
                print("done in %0.3fs" % (time() - t0))
                print()
                print("Best parameters: ")
                print(svc.best_params_)
                print()
                print("Best estimator: ")
                print(svc.best_estimator_)
                print()
                print("Best score: ")
                print(svc.best_score_)
                print()
                # joblib.dump(svc, 'train_model_64_all_rbf_c8_' + str(index) + '.pkl', compress=True)

                # start prediction
                print("Started SVM prediction on test set ")
                t0 = time()
                filename = dataPath + testDataName + str(index) + '.mat'
                file = h5py.File(filename, 'r')

                # read testing data
                testData = np.transpose(np.array(file.get('data')))
                testLabels = np.transpose(np.array(file.get('labels')))

                # predict from test data
                predictedLabels = svc.predict(testData)
                print("done in %0.3fs" % (time() - t0))
                print()
                print("Accuracy Score: %f" % (100*accuracy_score(testLabels, predictedLabels)))
                print()
                print(classification_report(testLabels, predictedLabels))
                print()
                print(confusion_matrix(testLabels, predictedLabels, labels=range(1, 201)))
                print()

                if isLogging:
                    sys.stdout = old_stdout
                    log_file.close()