from os.path import dirname, abspath, join, sep
from Utility import Utility

absolutePath = dirname(dirname(dirname(abspath("__file__"))))
featureAnnotationsDir = join(absolutePath, 'feature-annotations')
featureName = "FV_d2_k128"  # str(sys.argv[1])
resultFileName = join(absolutePath, 'results')
resultFileName = join(resultFileName, 'Results_' + featureName + '.mat')

# prepare the result text file for classification
annotationToVideoMapDir = featureAnnotationsDir + sep + 'anonymous_aid.csv'
outputFileName = join(absolutePath, 'results' + sep + featureName + '_output.txt')
Utility.prepareTestSubmissionFile(resultFileName, annotationToVideoMapDir, outputFileName)