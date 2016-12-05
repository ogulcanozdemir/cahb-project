import csv
import h5py
import numpy as np
from os.path import join
from sklearn.preprocessing import MultiLabelBinarizer

class Utility:
    
    @staticmethod
    def readBaselineIDTFeatures(featureFile, trainAnnotations, testAnnotations):
        print('Reading baseline IDT features with train and test labels...')
        file = h5py.File(featureFile, 'r')
    
        # assign every label to it's corresponding action class using annotations
        matFiles = file['matFiles/name']
        videoIndexes = [u''.join(chr(c) for c in file[obj_ref]) for obj_ref in matFiles[0][0:20]] # FIXME : type casting is too slow
        videoIndexes = [x[:-15] for x in videoIndexes]

        features = file['FV']
        trainData = []
        trainLabels = []
        trainAnnotations_subsampled = []
        testData = []
        testLabels = []
        testAnnotations_subsampled = []

        for idx, video in enumerate(videoIndexes):
            result = Utility.findAnnotation(trainAnnotations, video)
            if result is not None and len(result) == 2:
                trainData.append(np.array(features[idx]))
                trainLabels.append(result[1])
                trainAnnotations_subsampled.append(video)
            else:
                result = Utility.findAnnotation(testAnnotations, video)
                if result is not None and len(result) == 2:
                    testData.append(np.array(features[idx]))
                    testLabels.append(result[1])
                    testAnnotations_subsampled.append(video)

        return np.array(trainData), trainLabels, trainAnnotations_subsampled, np.array(testData), testLabels, testAnnotations_subsampled
    
    @classmethod
    def __mapActionClasses(cls, row):
        actionClasses = [x[:4] for x in row.split(';')]
        classes = np.asarray(actionClasses)
        
        binArray = np.zeros(157)
        for c in classes:
            if c:
                binArray[int(c[1:])] = 1
            
        return binArray
        
    @staticmethod
    def readAnnotations(annotationFile):
        print('Reading annotation file : ' + annotationFile)
        data = []
        with open(annotationFile) as file:
            _ = next(file)
            reader = csv.reader(file)   
            for row in reader:
                #classes = MultiLabelBinarizer().fit_transform(row[11])
                classes = Utility.__mapActionClasses(row[11])
                rowDict = { 'aid' : row[0], 'classes':  classes}
                data.append(rowDict)
            del reader
        
        return data              
        
    @staticmethod
    def saveAnnotations(annotations, filename):
        print('Saving annotations to data/' + filename + '...')
        with open(join('data', filename), 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["aid", "classes"])
            for item in annotations:
                writer.writerow([item['aid'], item['classes']])
            del writer
            
    @staticmethod
    def findAnnotation(annotations, aid):
        for idx, d in enumerate(annotations):
            if d['aid'] == aid:
                return idx, d['classes'] 

    @staticmethod
    def findVideo(videoMap, annotation):
        for idx, vmap in enumerate(videoMap):
            if vmap[0] == annotation:
                return vmap[1]

    @staticmethod
    def prepareTestSubmissionFile(resultFileName, annotationToVideoMapDir, outputFileName):
        annotationToVideoMap = []
        with open(annotationToVideoMapDir) as file:
            _ = next(file)
            reader = csv.reader(file)
            for row in reader:
                annotationToVideoMap.append(row)
            del row, reader

        resultFile = h5py.File(resultFileName, 'r')
        resultLabels = np.transpose(resultFile['resultsLabels'])
        resultProba = np.transpose(resultFile['resultsProba'])
        testAnnotations = resultFile['testAnnotations']
        testAnnotations = [u''.join(chr(c) for c in resultFile[obj_ref]) for obj_ref in testAnnotations[0][:]]

        # prepare log probabilities
        resultLogProba = resultProba
        resultLogProba[resultProba == 0] = 1e-6
        resultLogProba = np.log(resultLogProba)

        results = []
        for idx, proba in enumerate(resultProba):
            video = Utility.findVideo(annotationToVideoMap, testAnnotations[idx])
            results.append(video + ' ' + ' '.join(str(s) for s in resultLogProba[idx]))

        with open(outputFileName, "w") as f:
            for item in results:
                f.write("%s\n" % item)