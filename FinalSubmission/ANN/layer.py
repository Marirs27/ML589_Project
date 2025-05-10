import numpy as np

class Layer:
    def __init__(self, inputSize, outputSize, l):
        self.a = np.zeros((inputSize,1))
        self.a = np.vstack((np.ones((1,1)),self.a)) # Adding Bias
        self.blame = np.zeros((inputSize+1,1))
        self.weight = np.zeros((outputSize, inputSize+1))
        self.gradient = np.zeros((outputSize, inputSize+1))
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.l = l

    def printA(self):
        print("a"+str(self.l)+": ")
        self.matrixPrint(self.getTranspose(self.a))
    
    def printBlame(self):
        print("delta"+str(self.l)+": ")
        self.matrixPrint(self.blame.T)
    
    def printWeight(self):
        print("Theta"+str(self.l)+": ")
        self.matrixPrint(self.weight)

    def printGradient(self, instanceGradientTracker={}):
        for layer in instanceGradientTracker.keys():
            print(f"Gradient {layer}")
            self.matrixPrint(instanceGradientTracker[layer])

    def matrixPrint(self, matrix):
        for row in matrix:
            print(row)
        print()
    
    def getTranspose(self, matrix):
        return np.transpose(matrix)
    