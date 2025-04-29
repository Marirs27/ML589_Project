'''
Implement the method (discussed in class) that allows you to numerically
check the gradients computed by backpropagation. This will allow you to further ensure that all gradients
computed by your implementation of backpropagation are correct; that way, you will be confident that you
trust your implementation of the neural network training procedure. You should present in your report
the estimated gradients for the two neural networks described in the provided benchmark files. First,
estimate the gradients using ε = 0.1, and then ε = 0.000001. If you choose to work on this extra-credit
task, please include the corresponding source code along with your Gradescope submission.

--------
Test 1:
--------
Initializing the network with the following structure (number of neurons per layer excluding bias): [1 2 1]

Initial Theta1 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.40000  0.10000  
	0.30000  0.20000  

Initial Theta2 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.70000  0.50000  0.60000  

Training set
	Training instance 1
		x: [0.13000]
		y: [0.90000]
Training instance 2
		x: [0.42000]
		y: [0.23000]


--------
Test 2:
--------
Regularization parameter lambda=0.250

Initializing the network with the following structure (number of neurons per layer): [2 4 3 2]

Initial Theta1 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.42000  0.15000  0.40000  
	0.72000  0.10000  0.54000  
	0.01000  0.19000  0.42000  
	0.30000  0.35000  0.68000  

Initial Theta2 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.21000  0.67000  0.14000  0.96000  0.87000  
	0.87000  0.42000  0.20000  0.32000  0.89000  
	0.03000  0.56000  0.80000  0.69000  0.09000  

Initial Theta3 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.04000  0.87000  0.42000  0.53000  
	0.17000  0.10000  0.95000  0.69000  


Training set
	Training instance 1
		x: [0.32000   0.68000]
		y: [0.75000   0.98000]
	Training instance 2
		x: [0.83000   0.02000]
		y: [0.75000   0.28000]
'''

import numpy as np
from backPropagation import BackPropagation
from layer import Layer
from forwardPropagation import ForwardPropagation

def errorDerivative(layers, x, y, weightIndex=(1,1,2), epsilon=0.000001, regularizationValue=0):
    print("Epsilon: ", epsilon)
    layers[weightIndex[0]-1].weight[weightIndex[1]][weightIndex[2]] += epsilon
    forwardPropagation1 = ForwardPropagation(layers=layers, batchSize=2, regularization=regularizationValue)
    for i in range(len(x)):
        x_instance = x[i]
        y_instance = y[i]
        forwardPropagation1.forward(x_instance)
        forwardPropagation1.y = y_instance
        forwardPropagation1.calculateError(i)
    forwardPropagation1.calculateAvgError()
    err1 = forwardPropagation1.J
    print(f"Error with theta + epsilon: {err1}")

    layers[weightIndex[0]-1].weight[weightIndex[1]][weightIndex[2]] -=2* epsilon
    forwardPropagation2 = ForwardPropagation(layers=layers, batchSize=2, regularization=regularizationValue)
    for i in range(len(x)):
        x_instance = x[i]
        y_instance = y[i]
        forwardPropagation2.forward(x_instance)
        forwardPropagation2.y = y_instance
        forwardPropagation2.calculateError(i)
    forwardPropagation2.calculateAvgError()
    err2 = forwardPropagation2.J
    print(f"Error with theta - epsilon: {err2}")
    print("Gradient error derivative: ", (err1 - err2) / (2 * epsilon))
    return (err1 - err2) / (2 * epsilon)

def actualErrorDerivative(layers, x, y, weightIndex=(1,1,2), regularizationValue=0):
    forwardPropagation = ForwardPropagation(layers=layers, batchSize=2, regularization=regularizationValue)
    backPropagation = BackPropagation(layers, batchSize=2, regularization=regularizationValue)
    for i in range(len(x)):
        x_instance = x[i]
        y_instance = y[i]
        forwardPropagation.forward(x_instance)
        backPropagation.y = y_instance
        backPropagation.calculateBlame()
        backPropagation.calculateGradient()
    backPropagation.calculateAvgGradient()
    print("Value of Gradient from backpropagation: ", layers[weightIndex[0]-1].gradient[weightIndex[1]][weightIndex[2]])
    return layers[weightIndex[0]-1].gradient[weightIndex[1]][weightIndex[2]]

def initTest1():
    layers = []
    inputLayer = Layer(neuronsPerLayer[0], neuronsPerLayer[1], l=1)
    inputLayer.weight = np.array([[0.4, 0.1], [0.3, 0.2]])
    layers.append(inputLayer)
    hiddenLayer = Layer(neuronsPerLayer[1], neuronsPerLayer[2], l=2)
    hiddenLayer.weight = np.array([[0.7, 0.5, 0.6]])
    layers.append(hiddenLayer)
    outputLayer = Layer(neuronsPerLayer[2], 0, l=3)
    layers.append(outputLayer)
    return layers

def initTest2():
    layers = []
    inputLayer = Layer(neuronsPerLayer[0], neuronsPerLayer[1], l=1)
    inputLayer.weight = np.array([[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]])
    layers.append(inputLayer)
    hiddenLayer1 = Layer(neuronsPerLayer[1], neuronsPerLayer[2], l=2)
    hiddenLayer1.weight = np.array([[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89], [0.03, 0.56, 0.8, 0.69, 0.09]])
    layers.append(hiddenLayer1)
    hiddenLayer2 = Layer(neuronsPerLayer[2], neuronsPerLayer[3], l=3)
    hiddenLayer2.weight = np.array([[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]])
    layers.append(hiddenLayer2)
    outputLayer = Layer(neuronsPerLayer[3], 0, l=4)
    layers.append(outputLayer)
    return layers


if __name__ == "__main__":
    test1 = True
    test2 = True
    if test1:
        print("\n------------------")
        print("Backpropagation 1")
        print("------------------")
        neuronsPerLayer = [1, 2, 1]
        x_train = [np.array([[0.13]]), np.array([[0.42]])]
        y_train = [np.array([[0.9]]), np.array([[0.23]])]

        layers = initTest1()
        errorDerivative(layers, x_train, y_train, weightIndex=(1,1,1), epsilon=0.00001)
        layers = initTest1()
        errorDerivative(layers, x_train, y_train, weightIndex=(1,1,1), epsilon=0.1)
        layers = initTest1()
        actualErrorDerivative(layers, x_train, y_train, weightIndex=(1,1,1))
    if test2:
        print("\n------------------")
        print("Backpropagation 2")
        print("------------------")
        x_train = [np.array([[0.32], [0.68]]), np.array([[0.83], [0.02]])]
        y_train = [np.array([[0.75], [0.98]]), np.array([[0.75], [0.28]])]
        neuronsPerLayer = [2, 4, 3, 2]

        layers = initTest2()
        errorDerivative(layers, x_train, y_train, weightIndex=(1,1,1), epsilon=0.00001,regularizationValue=0.25)
        layers = initTest2()
        errorDerivative(layers, x_train, y_train, weightIndex=(1,1,1), epsilon=0.1,regularizationValue=0.25)
        layers = initTest2()
        actualErrorDerivative(layers, x_train, y_train, weightIndex=(1,1,1),regularizationValue=0.25)
