# Data Pre processing module
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class PreProcesser:
    def __init__(self, data=None, normalized=True):
        self.data = data
        self.normalized = normalized

    def setData(self):
        print("DataFrame Columns:\n", self.data.columns)
        # self.data = sklearn_shuffle(self.data);, "label" is the y col seperate data based on col
        self.y = self.data["Diagnosis"].values
        self.X = self.data.drop("Diagnosis", axis=1).values
        
    def split(self,testPercent=20, random_state=None):
        return train_test_split(self.X,self.y, test_size=testPercent/100, random_state=random_state,shuffle=True)

    def normalize(self):
        if self.normalized:
            # Convert self.X to a DataFrame for processing
            self.X = pd.DataFrame(self.X)
            
            # Identify categorical columns (non-numeric)
            categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns
            
            # Apply OneHotEncoder to categorical columns
            if not categorical_columns.empty:
                encoder = OneHotEncoder(sparse_output=False)  # Use dense output
                encoded = encoder.fit_transform(self.X[categorical_columns])
                encoded_df = pd.DataFrame(encoded, index=self.X.index)
                
                # Drop original categorical columns and concatenate encoded columns
                self.X = pd.concat([self.X.drop(columns=categorical_columns), encoded_df], axis=1)
            
            # Ensure all data in self.X is numeric
            self.X = self.X.apply(pd.to_numeric, errors='coerce')
            
            # Check for NaN values and fill them with 0
            self.X = self.X.fillna(0)
            
            # Perform normalization
            self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
            self.X = self.X.to_numpy()

    def preprocess(self):
        self.setData();
        self.normalize()

    def save(self, file_name):
        # Combining X and y into a single DataFrame without header
        self.data = pd.concat([pd.DataFrame(self.X), pd.DataFrame(self.y)], axis=1)
        self.data.to_csv(file_name, index=False, header=False)
        print("Data Saved\n")
    
    def load(self,file_name):
        self.data = pd.read_csv(file_name, header=None)
        print("Data Loaded\n")
