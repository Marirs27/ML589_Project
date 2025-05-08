import os
import sys
import numpy as np
from dataProcess import DataPreprocessor
from trainModel import TrainModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'KNN')))
from KNN_Instance import KNNModel

class ModelSampler:
    EPSILON = 1e-6
    REGULARIZATION_VALUES = [0.01,0.025]
    STEP_SIZE_VALUES = [0.01, 0.05]
    BATCH_SIZE_VALUES = [10,32, 20]
    K_FOLD = 10
    EPOCHS = 100
    
    def __init__(self, filePath, splice = None, labelColumn = 'label'):
        self.filePath = filePath
        self.preprocessor = DataPreprocessor(filePath=filePath, kFold=self.K_FOLD, splice = None, randomSeed=42, labelColumn=labelColumn)
        self.preprocessor.load_data()
        self.preprocessor.encodeCategorical()
        self.preprocessor.normalizeData()
        self.preprocessor.stratifiedKFold()
        self.preprocessor.printDataDetails()
        self.trainModels = []
        self.accuracy = []
        self.f1Score = []
        self.confusionMatrix = []
        self.precision = []
        self.recall = []


    def sampleModels(self, layerSkeleton, regularization=0.01, stepSize=0.01, batchSize=10, thresholdValue=0.5, stoppingCriterionCategory='epochs'):
        # Store all the models and their metrics to plot
        accuracy = []
        f1Score = []
        precision = []
        recall = []
        loss = []
        modelAccuracy = []
        modelF1Score = []
        models = []
        modelDecEpoch = []
        model = None

        for layers in layerSkeleton:
            # Add input layer size to the beginning of the architecture
            l = layers.copy()
            l.insert(0, self.preprocessor.data.shape[1] - 1)
            print(self.preprocessor.data.shape[1] - 1, "---------------", self.preprocessor.data.shape)

            # Create and train the model
            model = TrainModel(self.preprocessor, l, self.EPSILON, batchSize, regularization=regularization, stepSize=stepSize, threshold=thresholdValue, epoch=self.EPOCHS)
            print("\n\n")
            print(f"Model with layers {l}, regularization {regularization}, batch size {batchSize}, step size {stepSize} created successfully")

            # Train the model using k-fold cross-validation and get Learning curve (metric vs epoch list)
            accLC, preLC, recLC, f1LC, lossLC = model.kFoldTrainTest(stoppingCriterion=stoppingCriterionCategory)

        print("Model sampling completed successfully")
        return model
    







if __name__ == "__main__":
    layerSkeletons = [
      [18, 10, 1]
    ]
    # modelSampler = ModelSampler(filePath='ANN/datasets/loan.csv')
    # for reg in modelSampler.REGULARIZATION_VALUES:
    #     for step in modelSampler.STEP_SIZE_VALUES:
    #         for batch in modelSampler.BATCH_SIZE_VALUES:
    #             print(f"Sampling models with regularization={reg}, stepSize={step}, batchSize={batch}")
    #             # layerSkeletons.extend(LOAN_LAYERS_SKELETON) 
    #             modelSampler.sampleModels(
    #                 layerSkeleton=LOAN_LAYERS_SKELETON,
    #                 regularization=reg,
    #                 stepSize=step,
    #                 batchSize=batch,
    #                 stoppingCriterionCategory='epochs'
    #             )
    #             print("Model sampling completed successfully")

    modelSampler = ModelSampler(filePath='datasets/parkinsons.csv', labelColumn='Diagnosis')
    modelSampler.EPOCHS = 100
    featureModel = modelSampler.sampleModels(
        layerSkeleton=layerSkeletons,
        regularization=0.027,
        stepSize=0.05,
        batchSize=25,
        stoppingCriterionCategory='epochs'
    )

    # Get train/test splits from your preprocessor
    X_train, y_train, X_test, y_test = modelSampler.preprocessor.getTrainTestSplit(0)

    # Extract features for each instance in train set using .forward()
    features_train = []
    for i in range(X_train.shape[0]):
        featureModel.forwardPropagation.forward(X_train[i].reshape(-1, 1))
        features_train.append(featureModel.forwardPropagation.layers[-2].a[1:].flatten())
    features_train = np.array(features_train)

    # Extract features for each instance in test set
    features_test = []
    for i in range(X_test.shape[0]):
        featureModel.forwardPropagation.forward(X_test[i].reshape(-1, 1))
        features_test.append(featureModel.forwardPropagation.layers[-2].a[1:].flatten())
    features_test = np.array(features_test)

    # Use your custom KNN model
    knn = KNNModel(k=10)
    knn.trainModel(features_train, y_train)
    y_pred = knn.testModel(features_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"KNN on ANN features - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

