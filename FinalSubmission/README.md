# ML589 Final Project

This repository contains implementations for various machine learning models and hybrid approaches for your final project. The structure is as follows:

```
FinalSubmission/
│
├── ANN/
│   ├── backPropagation.py
│   ├── calculateAcuracy.py
│   ├── dataProcess.py
│   ├── forwardPropagation.py
│   ├── hybridModel.py
│   ├── layer.py
│   ├── modelSampler.py
│   ├── trainModel.py
│
├── KNN/
│   ├── calculate_accuracy.py
│   ├── data_analysing.py
│   ├── data_preprocessing.py
│   ├── data.py
│   ├── KNN_Instance.py
│   ├── main.py
│   ├── model_sampler.py
│
├── Logistic Regression/
│   ├── logisticRegression.py
│
├── RandomForest/
│   ├── RandomForest.py
│
└── README.md
```

---

## 1. Artificial Neural Network (ANN)

**Location:** `ANN/`

### Main Files:
- `trainModel.py`: Contains the main ANN training logic.
- `modelSampler.py`: For sampling and evaluating different ANN architectures.

### How to Run ANN:

To sample and evaluate models:

```bash
python modelSampler.py
```

You can edit the `if __name__ == "__main__":` block in `modelSampler.py` to specify dataset, architecture, and parameters.

---

## 2. K-Nearest Neighbors (KNN)

**Location:** `KNN/`

### Main Files:
- `KNN_Instance.py`: Custom KNN implementation.
- `main.py`: Script to run KNN experiments.

### How to Run KNN:

```bash
cd FinalSubmission/KNN
python main.py
```

You can modify `main.py` to change dataset paths or K values.

---

## 3. Logistic Regression

**Location:** `Logistic Regression/`

### Main Files:
- `logisticRegression.py`: Custom logistic regression implementation.

### How to Run Logistic Regression:

```bash
cd "FinalSubmission/Logistic Regression"
python logisticRegression.py
```

---

## 4. Random Forest

**Location:** `RandomForest/`

### Main Files:
- `RandomForest.py`: Random Forest implementation and experiment script.

### How to Run Random Forest:

```bash
cd FinalSubmission/RandomForest
python RandomForest.py
```

---

## 5. Hybrid Model (ANN + KNN)

**Location:** `ANN/hybridModel.py`

This script combines ANN feature extraction with KNN classification and supports hyperparameter tuning.

### How to Run the Hybrid Model:

```bash
cd FinalSubmission/ANN
python hybridModel.py
```

By default, it will run the main experiment as defined in the `if __name__ == "__main__":` block.  
You can edit this block to change the dataset, architecture, or parameters.

---

## 6. Hybrid Model Tuning

**Function:** `hybrid_hyperparam_tuning` in `ANN/hybridModel.py`

### How to Run Hyperparameter Tuning

To run hyperparameter tuning for the hybrid model, **uncomment and edit** the following block at the bottom of `hybridModel.py`:

```python
hybrid_hyperparam_tuning(
    filePath='datasets/rice.csv',
    labelColumn='label',
    layerSkeletons=[[3,4,18,1]],
    regularizations=[0.01, 0.025],
    stepSizes=[0.01, 0.05],
    batchSizes=[10, 32],
    knn_ks=[5, 10, 15, 20]
)
```

Then run:

```bash
python hybridModel.py
```

This will perform grid search over the specified hyperparameters and save results to `hybrid_hyperparam_results.csv`.

---

## 7. Notes

- **Datasets:** Place your datasets (e.g., `rice.csv`, `credit_approval.csv`, `digits.csv`) in a `datasets/` folder at the root or update the file paths in the scripts.
- **Dependencies:** Install required packages with:
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn
  ```


---
