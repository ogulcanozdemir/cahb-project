from joblib import Parallel, delayed
import multiprocessing
import os
from glob import glob
import subprocess
import gzip

dirPath = os.path.dirname(os.path.realpath(__file__))
videoPath = dirPath + os.sep + 'charades-data' + os.sep
featurePath = dirPath + os.sep + 'charades-features' + os.sep
executable = dirPath + os.sep + 'DenseTrackStab'


def extract(idx, filePath):
    print str(idx) + " : " + filePath
    fileName = os.path.basename(filePath)
    sp = subprocess.Popen([executable, filePath], stdout=subprocess.PIPE)
    dest = featurePath + fileName[:-4] + '.features.gz'
    with gzip.open(dest, 'wb') as f:
        f.write(sp.stdout.read())


def getFinishedVideoNames():
    finishedTxt = 'first_charades_features.txt'
    finishedVideoNames = []
    for line in open(finishedTxt):
        finishedVideoNames.append(line[-18:-13])
    return finishedVideoNames[2:]  # Ignore `.` and `..` case.


if __name__ == '__main__':
    finishedVideoNames = getFinishedVideoNames()
    videoPathList = glob(videoPath + '*.mp4')
    unfinishedVideoPathList = []
    for videoPath in videoPathList:
        if videoPath[-9:-4] not in finishedVideoNames:
            unfinishedVideoPathList.append(videoPath)
    numCores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=numCores)(
        delayed(extract)(idx, name) for idx, name in enumerate(unfinishedVideoPathList))
