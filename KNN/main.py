import pandas as pd
from sklearn.calibration import LabelEncoder
from model_sampler import KKNSampler, plotComparision
from data_preprocessing import PreProcesser
from data_analysing import Analyser
from KNN_Instance import KNNModel
from calculate_accuracy import CalculateAccuracy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import fetch_openml, load_iris, load_wine

# file_path = 'datasets/parkinsons.csv'
# df = pd.read_csv(file_path, header=None)

#  Load the digits dataset
# digits = datasets.load_digits(return_X_y=True)
# digits_dataset_X = digits[0]
# digits_dataset_y = digits[1]
# N = len(digits_dataset_X)
# data = pd.DataFrame(digits_dataset_X, columns=[f'pixel_{i}' for i in range(digits_dataset_X.shape[1])])
# data['label'] = digits_dataset_y
# df = data.copy()

# Load the Kuzushiji-MNIST (Japanese letters) dataset
kuzushiji = fetch_openml('Kuzushiji-MNIST', version=1, as_frame=False)
X = kuzushiji.data/ 255.0
y = kuzushiji.target.astype(int)  # Already numeric labels 0-9
X, y = X[:1000], y[:1000]

data = pd.DataFrame(X, columns=[f"feature{i}" for i in range(X.shape[1])])
data["Diagnosis"] = y
df = data.copy()


# Pre process
num_columns = df.shape[1]
column_names = [f"feature{i}" for i in range(num_columns - 1)] + ["Diagnosis"]
df.columns = column_names

# Create Processor Object
wbdcProcessor = PreProcesser(df)
wbdcAnalyser = Analyser(df)
# knnSamplerTrainN = KKNSampler(df,k_range=range(10, 30, 2), test_data=False, sampling_runs=20)
knnSamplerTestN = KKNSampler(df,k_range=range(2, 20, 2), test_data=True, sampling_runs=20)
# knnSamplerTrain = KKNSampler(df,k_range=range(1, 12, 2), test_data=False, sampling_runs=20,normalized=False)
# knnSamplerTest = KKNSampler(df,k_range=range(1, 12, 2), test_data=True, sampling_runs=20,normalized=False)

'''
    Data Analysing
'''
#Create Analyser Object
# wbdcAnalyser.info()
# wbdcAnalyser.calculate_A_Priori()
# wbdcAnalyser.plot()
# wbdcAnalyser.plot_pie()

'''
    KNN Classifier
'''
# Run sampler for Normalized data
# knnSamplerTrainN.run()
# knnSamplerTrainN.plotModel()
knnSamplerTestN.run()
knnSamplerTestN.plotModel()

# Run sampler for Non-Normalized data
# knnSamplerTrain.run()
# knnSamplerTrain.plot(False)
# knnSamplerTest.run()
# knnSamplerTest.plot(False)


# plotComparision(knnSamplerTestN,knnSamplerTest,
#                 'With Normalization', 'Without Normalization',
#                 'Comparing with and without normalizing test data')
