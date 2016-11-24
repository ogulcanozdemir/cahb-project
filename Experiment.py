from os.path import dirname, abspath, join
from Utility import Utility

# initialize paths for reading process
absolutePath = dirname(dirname(dirname(abspath("__file__"))))
featureAnnotationsDir = join(absolutePath, 'feature-annotations')

# read feature annotations of charades dataset
trainAnnotationsFile = featureAnnotationsDir + '\charades_v04_train.csv'
testAnnotationsFile = featureAnnotationsDir + '\charades_v04_test.csv'

trainAnnotations = Utility.readAnnotations(trainAnnotationsFile)
testAnnotations = Utility.readAnnotations(testAnnotationsFile)

Utility.saveAnnotations(trainAnnotations, 'trainLabels.csv')
Utility.saveAnnotations(testAnnotations, 'testLabels.csv')
del featureAnnotationsDir, trainAnnotationsFile, testAnnotationsFile

# read baseline features
baselineFeaturesDir = join(absolutePath, 'baseline-features')

# read baseline IDT features which were extracted from charades dataset
