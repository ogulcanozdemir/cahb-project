from os.path import dirname, abspath, join, sep
from Utility import Utility

absolutePath = dirname(dirname(dirname(abspath("__file__"))))
featureAnnotationsDir = join(absolutePath, 'feature-annotations')
featureName = "FV_d3_k128"  # str(sys.argv[1])
resultFileName = 'elm_' + featureName + '.mat'

# prepare the result text file for classification
annotationToVideoMapDir = featureAnnotationsDir + sep + 'anonymous_aid.csv'
outputFileName = 'results_' + featureName + '_output.txt'
Utility.prepareTestSubmissionFile(resultFileName, annotationToVideoMapDir, outputFileName)