import csv
import h5py
import numpy as np
from os.path import join


class Utility:
    
    @staticmethod
    def readBaselineIDTFeatures(featureFile: str, trainAnnotations: dict, testAnnotations: dict):
        file = h5py.File(featureFile, 'r')
    
        #data = np.array(features.get('FV'))

        # assign every label to its corresponding action class using annotations
        matFiles = file.get('matFiles')        
        
        
        
        return file
        
        
    @staticmethod
    def readAnnotations(annotationFile: str) -> set:
        data = []
        with open(annotationFile) as file:
            _ = next(file)
            reader = csv.reader(file)   
            for row in reader:
                #actionClasses = row[11].split(';')
                #rowDict = { 'aid' : , 'actionClasses':  }
                data.append({row[0] : row[11]})
            del reader
        
        return data              
            
    @staticmethod
    def saveAnnotations(annotations: list, filename: str):
        with open(join('data', filename), 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["aid", "actionClasses"])
            for item in annotations:
                writer.writerow([item['aid'], item['actionClasses']])
            del writer