import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(path, target_column=None, dataset_name="Dataset"):
    print(f"\n=== {dataset_name} Analysis ===")
    df = pd.read_csv(path)

    # Basic info
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nStatistical Summary:")
    print(df.describe(include='all'))

    print("\nMissing Values:")
    print(df.isnull().sum())

    if target_column and target_column in df.columns:
        print("\nClass Distribution:")
        print(df[target_column].value_counts())
        
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=target_column)
        plt.title(f"{dataset_name} - Class Distribution")
        plt.show()

    # Correlation heatmap (numerical features only)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"{dataset_name} - Correlation Matrix")
    plt.show()

    # Histograms of features
    df.hist(bins=20, figsize=(15, 10))
    plt.suptitle(f"{dataset_name} - Feature Distributions")
    plt.tight_layout()
    plt.show()

# Paths to datasets
credit_path = "datasets/credit_approval.csv"
parkinsons_path = "datasets/parkinsons.csv"
rice_path = "datasets/rice.csv"

# Analyze each dataset
analyze_dataset(credit_path, target_column="Class", dataset_name="Credit Approval")
analyze_dataset(parkinsons_path, target_column="status", dataset_name="Parkinsons")
analyze_dataset(rice_path, target_column="Class", dataset_name="Rice")
