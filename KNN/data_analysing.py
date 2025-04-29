import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from importlib import reload
import matplotlib.pyplot as plt
import seaborn as sns

class Analyser:
    def __init__(self,data=None):
        self.data = data;

    def load(self,file_path):
        self.data = pd.read_csv(file_path)

    def info(self):
        print(self.data.info())
        print(self.data.describe())
    
    def calculate_A_Priori(self):
        total = len(self.data)
        label_0 = len(self.data[self.data.iloc[:,-1]== 0])
        label_1 = len(self.data[self.data.iloc[:,-1]== 1])
        print("A Priori Probabilities")
        print(f"Label 0: {label_0/total}")
        print(f"Label 1: {label_1/total}")
    
    def plot(self):
        plt.figure(figsize=(10,6))
        sns.heatmap(self.data.corr(),annot=True)
        plt.show()
    
    def plot_pie(self):
        label_distribution = self.data.iloc[:,-1].value_counts()
        plt.figure(figsize=(5,5))
        plt.pie(label_distribution,labels=label_distribution.index,autopct='%1.1f%%')
        plt.title("Label Distribution")
        plt.show()