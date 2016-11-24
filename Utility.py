import csv
from os.path import join

class Utility:
    
    @staticmethod
    def readBaselineFeatures():        
        return 0
        
    @staticmethod
    def readAnnotations(annotationFile: str) -> dict:
        data = []
        with open(annotationFile) as file:
            _ = next(file)
            reader = csv.reader(file)   
            for row in reader:
                actionClasses = row[11].split(';')
                rowDict = { 'aid' : row[0], 'actionClasses': actionClasses }
                data.append(rowDict)
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