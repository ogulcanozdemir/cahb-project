from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle


from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer

import numpy as np
from time import time

if __name__ == '__main__':
    X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
    y2 = shuffle(y1, random_state=1)
    y3 = shuffle(y1, random_state=2)
    Y = np.vstack((y1, y2, y3)).T
    n_samples, n_features = X.shape # 10,100
    n_outputs = Y.shape[1] # 3
    n_classes = 3
    
    t0 = time()
    
    
    
    
    forest = RandomForestClassifier(n_estimators=100, random_state=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    multi_target_forest.fit(X, Y)
    print("Training finished in %0.3fs" % (time() - t0))

    t0 = time()
    predictedLabels = multi_target_forest.predict(X)
    print("Testing finished in %0.3fs" % (time() - t0))
    print(predictedLabels)
