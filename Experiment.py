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


import h5py
import numpy as np

file = h5py.File(baselineFeaturesDir + '\FV_d3_k64.mat', 'r')
    
#data = np.array(features.get('FV'))

# assign every label to it's corresponding action class using annotations
matFiles = file['matFiles/name']
names = [u''.join(chr(c) for c in file[obj_ref]) for obj_ref in matFiles[0][:]] # FIXME : converting types is too slow
names = [x[:-15] for x in names]




# read baseline IDT features which were extracted from charades dataset
#features = Utility.readBaselineIDTFeatures(baselineFeaturesDir + '\FV_d3_k64.mat')
