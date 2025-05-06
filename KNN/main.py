import pandas as pd
from model_sampler import KKNSampler, plotComparision
from data_preprocessing import PreProcesser
from data_analysing import Analyser
from KNN_Instance import KNNModel
from calculate_accuracy import CalculateAccuracy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

file_path = 'datasets/parkinsons.csv'
df = pd.read_csv(file_path, header=None)

# Load the digits dataset
digits = datasets.load_digits(return_X_y=True)
digits_dataset_X = digits[0]
digits_dataset_y = digits[1]
N = len(digits_dataset_X)
data = pd.DataFrame(digits_dataset_X, columns=[f'pixel_{i}' for i in range(digits_dataset_X.shape[1])])
data['label'] = digits_dataset_y
df =data.copy()



num_columns = df.shape[1]
column_names = [f"feature{i}" for i in range(num_columns - 1)] + ["Diagnosis"]
df.columns = column_names

# Create Processor Object
wbdcProcessor = PreProcesser(df)
wbdcAnalyser = Analyser(df)
# knnSamplerTrainN = KKNSampler(df,k_range=range(10, 30, 2), test_data=False, sampling_runs=20)
# knnSamplerTestN = KKNSampler(df,k_range=range(10, 30, 2), test_data=True, sampling_runs=20)
# knnSamplerTrain = KKNSampler(df,k_range=range(1, 12, 2), test_data=False, sampling_runs=20,normalized=False)
knnSamplerTest = KKNSampler(df,k_range=range(1, 12, 2), test_data=True, sampling_runs=20,normalized=False)

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
# knnSamplerTrainN.plot()
# knnSamplerTestN.run()
# knnSamplerTestN.plot()

# Run sampler for Non-Normalized data
# knnSamplerTrain.run()
# knnSamplerTrain.plot(False)
knnSamplerTest.run()
knnSamplerTest.plot(False)


# plotComparision(knnSamplerTestN,knnSamplerTest,
#                 'With Normalization', 'Without Normalization',
#                 'Comparing with and without normalizing test data')
