import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

# Load the dataset
df = pd.read_csv("ANN/datasets/loan.csv")  # Make sure this file is in your working directory

# Separate features and target
X = df.drop("label", axis=1)
y = df["label"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_cols)
    ]
)

# Fit and transform
X_processed = preprocessor.fit_transform(X)

# Holdout train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Define the neural network: architecture (5,10) with 1 output neuron
mlp = MLPClassifier(hidden_layer_sizes=(5, 10), max_iter=1, warm_start=True, random_state=42)

# Collect loss vs training size
train_sizes = []
test_losses = []

for i in range(10, len(X_train), 5):
    mlp.fit(X_train[:i], y_train[:i])
    y_proba = mlp.predict_proba(X_test)
    loss = log_loss(y_test, y_proba)
    train_sizes.append(i)
    test_losses.append(loss)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, test_losses, marker='o', linestyle='-', color='darkblue')
plt.title("Test Loss vs Training Samples (Holdout Method)")
plt.xlabel("Number of Training Samples")
plt.ylabel("Test Loss (Log Loss)")
plt.grid(True)
plt.tight_layout()
plt.show()
