import csv
import os
import scipy.io
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

annotationDir = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'feature-annotations'

trainingAnnotationPath = annotationDir + os.sep + 'Charades_v1_train.csv'
testAnnotationPath = annotationDir + os.sep + 'Charades_v1_test.csv'

actionSet = set()

trainingLabels = {}

with open(trainingAnnotationPath) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        labels = []
        actions = row['actions'].split(';')
        for action in actions:
            if action[:4]:
                labels.append(int(action[1:4]))

        trainingLabels[row['id']] = labels

with open(testAnnotationPath) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        labels = []
        actions = row['actions'].split(';')
        for action in actions:
            if action[:4]:
                labels.append(int(action[1:4]))

        trainingLabels[row['id']] = labels

mat = scipy.io.loadmat('fv.mat')
fisherVectors = mat['fisherVectors']
videoNames = []
for name in mat['videoNames'].flatten():
    videoNames.append(name[0])

trainingLabelList = []

for name in videoNames:
    trainingLabelList.append(sorted(list(set(trainingLabels[name]))))
print(trainingLabelList[0:2])

binarizer = MultiLabelBinarizer()
binarizedLabels = binarizer.fit_transform(trainingLabelList[2:])

predictions = OneVsRestClassifier(LinearSVC(random_state=0)).fit(fisherVectors[2:, :], binarizedLabels) \
    .predict(fisherVectors[0:2, :])

predictedLabels = binarizer.inverse_transform(predictions)
