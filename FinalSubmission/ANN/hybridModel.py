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
import pandas as pd

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
            # plotLearningCurve(accLC, f1LC, preLC, recLC, title="Model Learning Curve of {} with architecture {} regularization={}, stepSize={}, batchSize={}".format(self.filePath.split('/')[2],l,regularization, stepSize, batchSize))
            

            plt.figure(figsize=(12, 6))
            plt.plot(accLC, label='Accuracy', color='blue')
            plt.plot(f1LC, label='F1 Score', color='orange')
            # plt.plot(lossLC, label='Loss', color='red')
            plt.title(f"Metrics for Model with layers {l}")
            plt.xlabel('Epochs')
            plt.ylabel('Metric')
            plt.legend()
            plt.grid()

        print("Model sampling completed successfully")
        return model
    

def calculate_metrics(y_true, y_pred):
    tp, fp, fn, tn = 0, 0, 0, 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            if true == 1:
                tp += 1
            else:
                tn += 1
        else:
            if true == 1:
                fn += 1
            else:
                fp += 1
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return accuracy, precision, recall, f1

def hybrid_hyperparam_tuning(
    filePath,
    labelColumn,
    layerSkeletons,
    regularizations=[0.01, 0.025],
    stepSizes=[0.01, 0.05],
    batchSizes=[10, 32],
    knn_ks=[5, 10, 15]
):

    results = []
    for reg in regularizations:
        for step in stepSizes:
            for batch in batchSizes:
                # Train ANN
                modelSampler = ModelSampler(filePath=filePath, labelColumn=labelColumn)
                modelSampler.EPOCHS = 100
                featureModel = modelSampler.sampleModels(
                    layerSkeleton=layerSkeletons,
                    regularization=reg,
                    stepSize=step,
                    batchSize=batch,
                    stoppingCriterionCategory='epochs'
                )
                X_train, y_train, X_test, y_test = modelSampler.preprocessor.getTrainTestSplit(0)
                # Extract ANN features
                features_train = []
                for i in range(X_train.shape[0]):
                    featureModel.forwardPropagation.forward(X_train[i].reshape(-1, 1))
                    features_train.append(featureModel.forwardPropagation.layers[-2].a[1:].flatten())
                features_train = np.array(features_train)
                features_test = []
                for i in range(X_test.shape[0]):
                    featureModel.forwardPropagation.forward(X_test[i].reshape(-1, 1))
                    features_test.append(featureModel.forwardPropagation.layers[-2].a[1:].flatten())
                features_test = np.array(features_test)
                # KNN tuning
                for k in knn_ks:
                    knn = KNNModel(k=k)
                    knn.trainModel(features_train, y_train)
                    y_pred = knn.testModel(features_test)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    print(f"reg={reg}, step={step}, batch={batch}, k={k} | Acc={acc:.4f}, F1={f1:.4f}")
                    results.append({
                        "regularization": reg,
                        "stepSize": step,
                        "batchSize": batch,
                        "knn_k": k,
                        "accuracy": acc,
                        "f1": f1
                    })
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("hybrid_hyperparam_results.csv", index=False)
    print("Saved results to hybrid_hyperparam_results.csv")
    # Print best
    best = results_df.loc[results_df['f1'].idxmax()]
    print("Best params:", best)
    return results_df




if __name__ == "__main__":
    layerSkeletons = [
      [3,4,18,1] # Your best ANN architecture
    ]

    # Change dataset accordingly to run the hybrid model
    modelSampler = ModelSampler(filePath='datasets/rice.csv', labelColumn='label')
    modelSampler.EPOCHS = 200
    featureModel = modelSampler.sampleModels(
        layerSkeleton=layerSkeletons,
        regularization=0.01,
        stepSize=0.01,
        batchSize=10,
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

    a = []
    f = []
    # Use your custom KNN model with different k values
    for i in range(7, 40, 2):
        knn = KNNModel(k=i)
        knn.trainModel(features_train, y_train)
        y_pred = knn.testModel(features_test)

        acc, _, _, f1_score_val = calculate_metrics(y_test, y_pred)
        a.append(acc)
        f.append(f1_score_val)
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(range(7,40, 2), a, yerr=[0.01]*len(a), fmt='o', capsize=5,
                 color='skyblue', ecolor='blue', elinewidth=0.8, capthick=1, markersize=6, label='Accuracy')
    plt.plot(range(7,40, 2), a, linestyle='-', marker='o', color='skyblue')
    plt.errorbar(range(7,40, 2), f, yerr=[0.01]*len(f), fmt='o', capsize=5,
                 color='salmon', ecolor='red', elinewidth=0.8, capthick=1, markersize=6, label='F1 Score')
    plt.plot(range(7,40, 2), f, linestyle='-', marker='o', color='salmon')
    for i, (k, acc, f1) in enumerate(zip(range(7,40, 2), a, f)):
        plt.annotate(f'{acc:.4f}', (k, acc), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
        plt.annotate(f'{f1:.4f}', (k, f1), textcoords="offset points", xytext=(0,-15), ha='center', color='red')
    plt.xlabel('K value')
    plt.ylabel('Score (%)')
    plt.grid()
    plt.legend()
    plt.title('Accuracy and F1 Score vs K value')
    plt.show()
    print(f"KNN on ANN features - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")


    # hybrid_hyperparam_tuning(
    #     filePath='datasets/rice.csv',
    #     labelColumn='label',
    #     layerSkeletons=layerSkeletons,
    #     regularizations=[0.01, 0.025],
    #     stepSizes=[0.01, 0.05],
    #     batchSizes=[10, 32],
    #     knn_ks=[5, 10, 15, 20]
    # )




