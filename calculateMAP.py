import csv
import os
import warnings
warnings.simplefilter('ignore', UserWarning)
import scipy.io
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score
from sklearn.externals import joblib

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

mat = scipy.io.loadmat('matlab' + os.sep + 'fv.mat')
fisherVectors = mat['fisherVectors']

videoNames = []
for name in mat['videoNames'].flatten():
    videoNames.append(name[0])

availableTrainingVideos = []
availableTestVideos = []
availableTrainingLabels = []
availableTestLabels = []
availableLabelNames = set()
trainingFishers = []
testFishers = []
for i, name in enumerate(videoNames):
    if name in testLabels:
        availableTestVideos.append(name)
        availableLabelNames |= set(testLabels[name])
        availableTestLabels.append(sorted(list(set(testLabels[name]))))
        testFishers.append(fisherVectors[i, :])

for i, name in enumerate(videoNames):
    if name in trainingLabels and len(set(trainingLabels[name]) - availableLabelNames) == 0:
        availableTrainingVideos.append(name)
        availableTrainingLabels.append(sorted(list(set(trainingLabels[name]))))
        trainingFishers.append(fisherVectors[i, :])

trainingFishers = np.array(trainingFishers)
testFishers = np.array(testFishers)

binarizer = MultiLabelBinarizer().fit(availableTestLabels)
binarizedLabels = binarizer.transform(availableTrainingLabels)

print('Number of actions: ', len(availableLabelNames))
print('Number of Training Videos: ', len(availableTrainingVideos))
print('Number of Test Videos: ', len(availableTestVideos))

confidenceScores = []

with open('confidenceScores.txt') as f:
    for line in f:
        confidenceScores.append([np.exp(float(s)) for s in line.split(' ')[1:]])
confidenceScores = np.array(confidenceScores)

# np.seterr(divide='ignore', invalid='ignore')
precision = average_precision_score(binarizer.transform(availableTestLabels), confidenceScores)
print('AP: ', precision)

