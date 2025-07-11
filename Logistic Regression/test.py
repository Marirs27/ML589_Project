import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from itertools import product

# --- Utility functions ---
def encode_categorical(X, y=None):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    X_df = pd.DataFrame(X)
    for col in X_df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col])
    if y is not None and (y.dtype == 'object' or str(y.dtype).startswith('category')):
        y = LabelEncoder().fit_transform(y)
    return X_df.values, y

def min_max_normalize(X):
    X = np.array(X, dtype=float)
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

# --- Load and preprocess data ---
credit_approval = pd.read_csv("datasets/credit_approval.csv")
y = credit_approval['label'].values
X = credit_approval.drop(columns=['label']).values

encode = True
if encode:
    X, y = encode_categorical(X, y)
X = min_max_normalize(X)

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# --- Hyperparameter grid ---
reg_lambdas = [0.001, 0.01, 0.1]
param_grid = {
    'C': [1/l for l in reg_lambdas],
    'solver': ['lbfgs'],
    'penalty': ['l2'],
    'multi_class': ['auto'],
}

# --- Manual grid search (no k-fold) ---
results = []
for C, solver, penalty, multi_class in product(param_grid['C'], param_grid['solver'], param_grid['penalty'], param_grid['multi_class']):
    model = LogisticRegression(C=C, solver=solver, penalty=penalty, multi_class=multi_class, max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({
        'C': C,
        'solver': solver,
        'penalty': penalty,
        'multi_class': multi_class,
        'accuracy': acc,
        'f1_score': f1
    })

# --- Find best ---
results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df['f1_score'].idxmax()]
print("Best Params:", best_row.to_dict())
print(f"Test Accuracy: {best_row['accuracy']:.4f}")
print(f"Test F1 Score: {best_row['f1_score']:.4f}")

# --- LaTeX table of all results ---
print(results_df.to_latex(index=False, float_format="%.4f"))