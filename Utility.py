import csv
import h5py
import numpy as np
from os.path import join

class Utility:
    
    @staticmethod
    def readBaselineIDTFeatures(featureFile, trainAnnotations, testAnnotations):
        print('Reading baseline IDT features with train and test labels...')
        file = h5py.File(featureFile, 'r')
    
        # assign every label to it's corresponding action class using annotations
        matFiles = file['matFiles/name']
        videoIndexes = [u''.join(chr(c) for c in file[obj_ref]) for obj_ref in matFiles[0][:]] # FIXME : type casting is too slow
        videoIndexes = [x[:-15] for x in videoIndexes]

        features = file['FV']
        trainData = []
        trainLabels = []
        testData = []
        testLabels = []

        for idx, video in enumerate(videoIndexes):
            result = Utility.findAnnotation(trainAnnotations, video)
            if result is not None and len(result) == 2:
                trainData.append(np.array(features[idx]))
                trainLabels.append(result[1])
            else:
                result = Utility.findAnnotation(testAnnotations, video)
                if result is not None and len(result) == 2:
                    testData.append(np.array(features[idx]))
                    testLabels.append(result[1])

        return np.array(trainData), trainLabels, np.array(testData), testLabels
    
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
                #actionClasses = row[11].split(';')
                #actionClasses = [x[:4] for x in row[11].split(';')]
                #classes = np.asarray(actionClasses)
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
