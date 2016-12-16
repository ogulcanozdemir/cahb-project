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

def extract(idx, file_path):
    print str(idx) + " : " + file_path;
    file_name = os.path.basename(file_path)
    sp = subprocess.Popen([executable, file_path], stdout=subprocess.PIPE)
    dest = featurePath + file_name[:-4] + '.features.gz'
    with gzip.open(dest, 'wb') as f:
        f.write(sp.stdout.read())

if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(extract)(idx, name) for idx, name in enumerate(glob(videoPath + '*.mp4')))
