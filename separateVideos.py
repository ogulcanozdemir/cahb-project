import csv
import os
import scipy.io
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score
from sklearn.externals import joblib

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

testLabels = {}
with open(testAnnotationPath) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        labels = []
        actions = row['actions'].split(';')
        for action in actions:
            if action[:4]:
                labels.append(int(action[1:4]))
        testLabels[row['id']] = labels


featureDir = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'matlab/charades-features'
trainingFeaturePath = featureDir+os.sep+'training'
testFeaturePath = featureDir+os.sep+'test'
if not os.path.exists(trainingFeaturePath):
    os.makedirs(trainingFeaturePath)
if not os.path.exists(testFeaturePath):
    os.makedirs(testFeaturePath)

for featureFileName in os.listdir(featureDir):
    videoName = featureFileName[:5]

# print(os.listdir(featureDir))

# annotationDir = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'feature-annotations'
# trainingAnnotationPath = annotationDir + os.sep + 'Charades_v1_train.csv'
# testAnnotationPath = annotationDir + os.sep + 'Charades_v1_test.csv'
#
# actionSet = set()
#
# trainingLabels = {}
# with open(trainingAnnotationPath) as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         labels = []
#         actions = row['actions'].split(';')
#         for action in actions:
#             if action[:4]:
#                 labels.append(int(action[1:4]))
#         trainingLabels[row['id']] = labels
#
# testLabels = {}
# with open(testAnnotationPath) as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         labels = []
#         actions = row['actions'].split(';')
#         for action in actions:
#             if action[:4]:
#                 labels.append(int(action[1:4]))
#         testLabels[row['id']] = labels
