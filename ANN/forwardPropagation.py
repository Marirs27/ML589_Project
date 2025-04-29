from layer import Layer
import numpy as np

class ForwardPropagation:
    def __init__(self, layers, y=None, batchSize=1, regularization=0):
        self.regularization = regularization
        self.batchSize = batchSize
        self.layers = layers
        self.J = 0
        self.y = y
        self.instanceErrorTracking= {}

    def g(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, x):
        self.layers[0].a = np.concatenate((np.array([[1]]), x), axis=0)
        for i in range(1, len(self.layers)):
            z = np.dot(self.layers[i - 1].weight, self.layers[i - 1].a)
            if i == len(self.layers) - 1:
                self.layers[i].a = self.g(z)
            else:
                self.layers[i].a = np.concatenate((np.array([[1]]), self.g(z)), axis=0)

    def printLayers(self):
        for layer in self.layers:
            layer.printA()
            layer.printWeight()

    def calculateError(self,instanceID):
        y = np.array(self.y)
        predictions = self.layers[-1].a 
        instanceError = -np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        self.instanceErrorTracking[instanceID] = instanceError
        self.J += instanceError
        return instanceError
    
    def calculateAvgError(self):
        self.J /= self.batchSize
        reg_term = 0
        for layer in self.layers[:-1]:
            reg_term += np.sum(np.square(layer.weight[:, 1:]))
        self.J += (self.regularization / (2 * self.batchSize)) * reg_term
