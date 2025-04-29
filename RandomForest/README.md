# Random Forest Classifier -HW 3


##  Function: `run_random_forest`
```python
def run_random_forest(data, attributes, ntree_values, k=5, max_depth=5, min_sample_split=2, min_info_gain=0.01):
```
### Parameters:

- `data`: `DataFrame` — Input dataset
- `attributes`: `list` — List of feature names
- `ntree_values`: `list[int]` — Values of number of trees to try (e.g. `[1, 5, 10]`)
- `k`: `int` — Number of folds for cross-validation (default=5)
- `max_depth`: `int` — Max depth of each decision tree
- `min_sample_split`: `int` — Minimum number of samples to allow a split
- `min_info_gain`: `float` — Minimum information gain required to split

Usage:
```python
run_random_forest("path to dataset.csv")
```

## Hyperparameter Tuning

### Function: `hyperparameter_tuning`
```python
hyperparameter_tuning(data, attributes)
```

### Parameters Searched:
- `min_samples_split`: [2, 5, 10]
- `max_depth`: [5, 10, 15]
- `min_info_gain`: [0.01, 0.05, 0.1]


Usage:
```python
data = load_data("dataset.csv")
attributes = [col for col in data.columns if col != "label"]

best_params = hyperparameter_tuning(data, attributes)
print("Best hyperparameters:", best_params)
```


##  File Structure

```
├── RandomForest.py
├── dataset.csv (all the datasets)
└── README.md
└── outputs
```