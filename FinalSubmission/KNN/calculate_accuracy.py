import numpy as np

class CalculateAccuracy:
    def __init__(self, y_test, y_pred):
        self.y_pred = y_pred
        self.y_test = y_test

    def accuracy(self):
        accuracy = np.mean(self.y_pred == self.y_test)
        return accuracy
    
    def accuracy_percentage(self):
        accuracy = self.accuracy()
        return accuracy * 100
        
