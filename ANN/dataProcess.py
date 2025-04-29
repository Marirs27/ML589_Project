import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self, filePath, labelColumn = 'label', kFold = 5, randomSeed = 42,):
        self.kFold = kFold
        self.randomSeed = randomSeed
        self.filePath = filePath
        self.data = None
        self.labelColumn = labelColumn

    def load_data(self):
        self.data = pd.read_csv(self.filePath)  # Load dataset with headers
        print(f"Data loaded successfully from {self.filePath}")
        print(f"Columns in the dataset: {self.data.columns}")
        
        # # Strip leading/trailing spaces from column names
        # self.data.columns = self.data.columns.str.strip()
        
        # Check if the label column exists
        if self.labelColumn not in self.data.columns:
            raise KeyError(f"Label column '{self.labelColumn}' not found in the dataset.")
    
    def encodeCategorical(self):
        # Encode the label column separately if it is categorical
        if self.data[self.labelColumn].dtype == 'object' or self.data[self.labelColumn].dtype.name == 'category':
            label_encoder = pd.get_dummies(self.data[self.labelColumn], drop_first=True)
            self.data = self.data.drop(columns=[self.labelColumn])
            self.data[self.labelColumn] = label_encoder

        # Identify other categorical columns (excluding the label column)
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns

        # Apply one-hot encoding to the remaining categorical columns
        self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)
        print("Categorical variables encoded successfully")
    
    def normalizeData(self):
        numeric_columns = self.data.select_dtypes(include=[np.number])
        self.data[numeric_columns.columns] = (numeric_columns - numeric_columns.min()) / (numeric_columns.max() - numeric_columns.min())
        print("Data normalized successfully")

    def stratifiedKFold(self): 
        np.random.seed(self.randomSeed)
        self.data = self.data.sample(frac=1, random_state=self.randomSeed).reset_index(drop=True)
        labels = self.data[self.labelColumn].values
        uniqueLabels = np.unique(labels)
        self.foldedDataIndex = [[] for _ in range(self.kFold)]
        for label in uniqueLabels:
            labelData = self.data[self.data[self.labelColumn] == label]
            labelDataIndices = labelData.index.tolist()
            foldSize = len(labelData) // self.kFold
            for i in range(self.kFold):
                start = i * foldSize
                end = (i + 1) * foldSize if i != self.kFold - 1 else len(labelData)
                self.foldedDataIndex[i].extend(labelDataIndices[start:end])
        return self.foldedDataIndex
    

    def getTrainTestSplit(self, k):
        # combine i index as test and others as train
        trainFolds = np.concatenate([np.array(self.foldedDataIndex[i]) for i in range(self.kFold) if i != k])
        testFold = np.array(self.foldedDataIndex[k])
        #'label' class is y and rest is x
        X_train = self.data.iloc[trainFolds].drop(columns=[self.labelColumn]).values  # Features
        y_train = self.data.iloc[trainFolds][self.labelColumn].values  # Labels
        X_test = self.data.iloc[testFold].drop(columns=[self.labelColumn]).values  # Features
        y_test = self.data.iloc[testFold][self.labelColumn].values  # Labels
        return X_train, y_train, X_test, y_test

    def printDataDetails(self):
        print(f"Data shape: {self.data.shape}")
        print(f"Label distribution:\n{self.data[self.labelColumn].value_counts()}")
        print(f"Data has been split into {self.kFold} folds")
        for i, fold in enumerate(self.foldedDataIndex):
            print(f"\tFold {i+1}: {len(fold)} samples")

if __name__ == "__main__":
    preprocessor = DataPreprocessor(filePath='ANN/datasets/raisin.csv')
    preprocessor.load_data()
    preprocessor.encodeCategorical()
    preprocessor.normalizeData()
    folds = preprocessor.stratifiedKFold()
    preprocessor.printDataDetails()



