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

mat = scipy.io.loadmat('matlab' + os.sep + 'fv128.mat')
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

classifier = OneVsRestClassifier(LinearSVC(random_state=0, verbose=3))
classifier.fit(trainingFishers, binarizedLabels)

joblib.dump(classifier, 'trainedLSVC128.pkl')

predictedLabels = classifier.predict(testFishers)
confidenceScores = classifier.decision_function(testFishers)

with open('confidenceScores128.txt', 'w') as f:
    for i, row in enumerate(confidenceScores):
        f.write(availableTestVideos[i] + ' ')
        f.write(' '.join([str(s) for s in row]))
        f.write(os.linesep)

precision = average_precision_score(binarizer.transform(availableTestLabels), confidenceScores)
print('==========')
print('Number of actions: ', len(availableLabelNames))
print('Number of Training Videos: ', len(availableTrainingVideos))
print('Number of Test Videos: ', len(availableTestVideos))
print('AP: ', precision)

# with open(testAnnotationPath) as inCsvfile, open('test.csv', 'a') as outCsvfile:
#     reader = csv.DictReader(inCsvfile)
#     writer = csv.DictWriter(outCsvfile, fieldnames=reader.fieldnames)
#     writer.writeheader()
#     for row in reader:
#         if row['id'] in availableTestVideos:
#             writer.writerow(row)
