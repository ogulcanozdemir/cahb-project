import csv
import os

annotationDir = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'feature-annotations'
trainingAnnotationPath = annotationDir + os.sep + 'Charades_v1_train.csv'
testAnnotationPath = annotationDir + os.sep + 'Charades_v1_test.csv'

trainingVideos = set()
with open(trainingAnnotationPath) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trainingVideos.add(row['id'])

testVideos = set()
with open(testAnnotationPath) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        testVideos.add(row['id'])

featureDir = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'matlab/charades-features'
trainingFeaturePath = featureDir + os.sep + 'training'
testFeaturePath = featureDir + os.sep + 'test'
if not os.path.exists(trainingFeaturePath):
    os.makedirs(trainingFeaturePath)
if not os.path.exists(testFeaturePath):
    os.makedirs(testFeaturePath)

for featureFileName in os.listdir(featureDir):
    src = featureDir + os.sep + featureFileName
    if os.path.isfile(src):
        videoName = featureFileName[:5]
        if videoName in trainingVideos:
            dest = trainingFeaturePath + os.sep + featureFileName
            os.rename(src, dest)
        elif videoName in testVideos:
            dest = testFeaturePath + os.sep + featureFileName
            os.rename(src, dest)
