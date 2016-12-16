from multiprocessing import Process
import os
from glob import glob
import subprocess
import gzip

dirPath = os.path.dirname(os.path.realpath(__file__))
videoPath = dirPath + os.sep + 'charades-data' + os.sep
featurePath = dirPath + os.sep + 'charades-features' + os.sep
executable = dirPath + os.sep + 'DenseTrackStab'


def extract(file_path):
    file_name = os.path.basename(file_path)
    sp = subprocess.Popen(['cat', file_path], stdout=subprocess.PIPE)
    dest = featurePath + file_name[:-4] + '.features.gz'
    with gzip.open(dest, 'wb') as f:
        f.write(sp.stdout.read())


if __name__ == '__main__':
    for file_path in glob(videoPath + '*.mp4'):
        p = Process(target=extract, args=(file_path,))
        p.start()
        p.join()
