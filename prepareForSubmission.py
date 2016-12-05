from os.path import dirname, abspath, join, sep
from Utility import Utility

absolutePath = dirname(dirname(dirname(abspath("__file__"))))
featureAnnotationsDir = join(absolutePath, 'feature-annotations')
featureName = "FV_d3_k128"  # str(sys.argv[1])
resultFileName = 'elm_' + featureName + '_tanh_nh100.mat'

# prepare the result text file for classification
annotationToVideoMapDir = featureAnnotationsDir + sep + 'anonymous_aid.csv'
outputFileName = 'results_' + featureName + '_tanh_nh100_output.txt'
Utility.prepareTestSubmissionFile(resultFileName, annotationToVideoMapDir, outputFileName)