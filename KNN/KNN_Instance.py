import numpy as np
from collections import Counter
import pandas as pd

class KNNModel:
    def __init__(self, k=5):
        self.k = k;

    def trainModel(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidian_dist(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)*(x1-x2)))

    def computeKfromEucledianDist(self,x):
        computedDistances = [self.euclidian_dist(x,xt) for xt in self.X_train]
        topKIndices = np.argsort(np.array(computedDistances))[:self.k]
        kPredictedNeighbours = [self.y_train.tolist()[i] for i in topKIndices]
        return Counter(kPredictedNeighbours).most_common(1)[0][0]

    def testModel(self,X_test):
        self.X_test = X_test
        result = [self.computeKfromEucledianDist(x) for x in self.X_test]
        return np.array(result)

    