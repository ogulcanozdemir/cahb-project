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
trainingFishers = []
testFishers = []
for i, name in enumerate(videoNames):
    if name in trainingLabels:
        availableTrainingVideos.append(name)
        availableTrainingLabels.append(sorted(list(set(trainingLabels[name]))))
        trainingFishers.append(fisherVectors[i, :])
    elif name in testLabels:
        availableTestVideos.append(name)
        availableTestLabels.append(sorted(list(set(testLabels[name]))))
        testFishers.append(fisherVectors[i, :])
trainingFishers = np.array(trainingFishers)
testFishers = np.array(testFishers)

binarizer = MultiLabelBinarizer()
availableTrainingLabels.append(list(range(157)))
binarizedLabels = binarizer.fit_transform(availableTrainingLabels)
del availableTrainingLabels[-1]
binarizedLabels = binarizedLabels[0:-1, :]
classifier = OneVsRestClassifier(LinearSVC(random_state=0))
classifier.fit(trainingFishers, binarizedLabels)

predictedLabels = classifier.predict(testFishers)

confidenceScores = classifier.decision_function(testFishers)

with open('confidenceScores.txt', 'a') as f:
    for i, row in enumerate(confidenceScores):
        f.write(availableTrainingVideos[i] + ' ')
        f.write(' '.join([str(s) for s in row]))
        f.write(os.linesep)

with open(testAnnotationPath) as inCsvfile, open('test.csv', 'a') as outCsvfile:
    reader = csv.DictReader(inCsvfile)
    writer = csv.DictWriter(outCsvfile,fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        if row['id'] in availableTestVideos:
            writer.writerow(row)


# f.write(content + os.linesep)

# predictions = OneVsRestClassifier(LinearSVC(random_state=0)).fit(fisherVectors[32:, :], binarizedLabels) \
#     .predict(fisherVectors[0:32, :])

# predictedLabels = binarizer.inverse_transform(predictions)

# print(predictedLabels)
