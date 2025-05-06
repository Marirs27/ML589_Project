import pandas as pd
from data_preprocessing import PreProcesser
from KNN_Instance import KNNModel
from calculate_accuracy import CalculateAccuracy
import numpy as np
import matplotlib.pyplot as plt

def stratified_kfold_indices(y, k=10, seed=42):
    """
    Returns a list of k lists, each containing the indices for one fold,
    stratified by the class distribution in y.
    """
    np.random.seed(seed)
    y = np.array(y)
    unique_labels = np.unique(y)
    folds = [[] for _ in range(k)]
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        np.random.shuffle(label_indices)
        fold_sizes = [len(label_indices) // k] * k
        for i in range(len(label_indices) % k):
            fold_sizes[i] += 1
        start = 0
        for i, size in enumerate(fold_sizes):
            folds[i].extend(label_indices[start:start+size])
            start += size
    return folds

class KKNSampler:
    def __init__(self,df, k_range: tuple, test_data=True, sampling_runs: int=20, normalized = True):
        self.df = df
        self.processorObj = PreProcesser(df, normalized)
        self.processorObj.preprocess()
        self.k_range = k_range
        self.sampling_runs = sampling_runs
        self.test_data = test_data
    
    def run(self,k_folds=10):
        # # We want to shuffle and split the data for each run (combination of k and sampling)
        # self.k_value = []
        # self.mean = []
        # self.std = []
        # for k in self.k_range:
        #     accuracies = [] # List to store accuracies for each sampling run
        #     for i in range(self.sampling_runs):
        #         print("Running K=",k," i=",i,"/20")
        #         # self.processorObj: PreProcesser = PreProcesser(self.df)
        #         # self.processorObj.shuffle()
        #         X_train, X_test, y_train, y_test = self.processorObj.split()
        #         # Now we want to trainåå the model and get accuracy
        #         knn = KNNModel(k=k)
        #         knn.trainModel(X_train, y_train)
        #         if self.test_data:
        #             predictions = knn.testModel(X_test)
        #             calcModel = CalculateAccuracy(y_test, predictions)
        #         else:
        #             predictions = knn.testModel(X_train)
        #             calcModel = CalculateAccuracy(y_train, predictions)
        #         # We use the Calculate Accuracy class to calculate the accuracy
        #         accuracies.append(calcModel.accuracy_percentage())
        #     meanK= np.mean(accuracies)
        #     stdK = np.std(accuracies)
        #     # We now get mean and standard deviations of the accuracies of k to plot
        #     self.mean.append(meanK)# mean of 20 rounds
        #     self.std.append(stdK) # std of 20 rounds
        #     self.k_value.append(k)
        # return self.mean,self.std,self.k_value
        self.k_value = []
        self.mean = []
        self.std = []
        y = self.df['Diagnosis'].values  # or whatever your label column is called
        folds = stratified_kfold_indices(y, k=k_folds, seed=42)
        for k in self.k_range:
            print("Running K=",k)
            accuracies = []
            for fold_idx in range(k_folds):
                test_idx = folds[fold_idx]
                train_idx = np.concatenate([folds[i] for i in range(k_folds) if i != fold_idx])
                X_train = self.df.iloc[train_idx].drop(columns=['Diagnosis']).values
                y_train = self.df.iloc[train_idx]['Diagnosis'].values
                X_test = self.df.iloc[test_idx].drop(columns=['Diagnosis']).values
                y_test = self.df.iloc[test_idx]['Diagnosis'].values
                knn = KNNModel(k=k)
                knn.trainModel(X_train, y_train)
                if self.test_data:
                    predictions = knn.testModel(X_test)
                    calcModel = CalculateAccuracy(y_test, predictions)
                else:
                    predictions = knn.testModel(X_train)
                    calcModel = CalculateAccuracy(y_train, predictions)
                predictions = knn.testModel(X_test)
                calcModel = CalculateAccuracy(y_test, predictions)
                accuracies.append(calcModel.accuracy_percentage())
            meanK = np.mean(accuracies)
            stdK = np.std(accuracies)
            self.mean.append(meanK)
            self.std.append(stdK)
            self.k_value.append(k)
        return self.mean, self.std, self.k_value
    
    def plotModel(self,normalized = True):
        colors = ('skyblue','blue') if self.test_data else ('salmon','red')
        plt.figure(figsize=(12,6))
        # We want to plot the k vs accuracy with error bars of std deviation cionnect the mean by line
        plt.errorbar(self.k_value, self.mean, yerr=self.std, fmt='o', capsize=5
                     ,color=colors[0], ecolor=colors[1], elinewidth=0.8, capthick=1, markersize=6)
        plt.plot(self.k_value, self.mean, linestyle='-', marker='o',color=colors[0])
        for i, (k,acc) in enumerate(zip(self.k_value,self.mean)):
            plt.annotate(f'{acc:.1f}', (self.k_value[i], self.mean[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('K value')
        plt.ylabel('Accuracy (%)')
        # plt.ylim(80, 100)  
        plt.grid()
        topic = 'Accuracy vs K value with Standard Deviation on'
        topic += ' Test Data' if self.test_data else ' Training Data'
        topic += ' - Normalized' if normalized else ' - Not Normalized'
        plt.title(topic)
        plt.show()


def plotComparision(knn1:KKNSampler,knn2:KKNSampler,label1,label2,title):
        color1, color2 = ('skyblue','blue'), ('salmon','red')
        plt.figure(figsize=(12,6))
        plt.figure(figsize=(12,6))
        # We want to plot the k vs accuracy with error bars of std deviation cionnect the mean by line
        plt.errorbar(knn1.k_value, knn1.mean, yerr=knn1.std, fmt='o', capsize=5
                     ,color=color1[0], ecolor=color1[1], elinewidth=0.8, capthick=1, markersize=6)
        plt.plot(knn1.k_value, knn1.mean, linestyle='-', marker='o',color=color1[0], label=label1)
        for i, (k,acc) in enumerate(zip(knn1.k_value,knn1.mean)):
            plt.annotate(f'{acc:.1f}', (knn1.k_value[i], knn1.mean[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.errorbar(knn2.k_value, knn2.mean, yerr=knn2.std, fmt='o', capsize=5
                     ,color=color2[0], ecolor=color2[1], elinewidth=0.8, capthick=1, markersize=6)
        plt.plot(knn2.k_value, knn2.mean, linestyle='-', marker='o',color=color2[0], label = label2)
        for i, (k,acc) in enumerate(zip(knn2.k_value,knn2.mean)):
            plt.annotate(f'{acc:.1f}', (knn2.k_value[i], knn2.mean[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('K value')
        plt.ylabel('Accuracy (%)')
        # plt.ylim(80, 100)  
        plt.legend()
        plt.grid()
        if knn1.test_data:
            dataset = 'Test Data'
        else:
            dataset = 'Training Data'
        plt.title('Accuracy vs K value with Standard Deviation for'+dataset +'- '+title)
        plt.show()