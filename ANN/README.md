
# Artificial Neural Networks - HW 2

## Main Function: TrainModel

### class TrainModel:

```python
def kFoldTrainTest(self, stoppingCriterion='epochs'):
```

#### Parameters:
- **stoppingCriterion**: `str` — Training stopping criterion ('epochs' or 'early_stopping')

#### Usage:

```python
from trainModel import TrainModel
from dataProcess import DataPreprocessor

# Example usage
preprocessor = DataPreprocessor(filePath='datasets/wdbc.csv', labelColumn='label')
preprocessor.load_data()
preprocessor.encodeCategorical()
preprocessor.normalizeData()
preprocessor.stratifiedKFold()

model = TrainModel(
    preprocessor,
    layersSkeleton=[30, 16, 8, 1],  # Example architecture
    epsilon=1e-6,
    batchSize=32,
    regularization=0.01,
    stepSize=0.05,
    threshold=0.5,
    epoch=100
)
model.buildModel()
model.kFoldTrainTest(stoppingCriterion='epochs')
```

Use stopping criteria as 'epochs' if you have epoch criteria, or if you want to stop the model with epsilon value diff in loss, use 'error'

---

## Hyperparameter Tuning

### Function: sampleModels

```python
modelSampler.sampleModels(
    layerSkeleton=[[30, 16, 8, 1]],
    regularization=0.01,
    stepSize=0.05,
    batchSize=32,
    stoppingCriterionCategory='epochs'
)
```

#### Parameters Searched:
- **layerSkeleton**: List of architectures to try (e.g. `[[30, 16, 8, 1], [30, 16, 1]]`)
- **regularization**: List of regularization strengths (e.g. `[0.01, 0.05, 0.1]`)
- **stepSize**: List of learning rates (e.g. `[0.01, 0.05, 0.1]`)
- **batchSize**: List of batch sizes (e.g. `[16, 32, 64]`)

---

## File Structure

```
trainModel.py
dataProcess.py
modelSampler.py
layer.py
forwardPropagation.py
backPropagation.py

datasets/
    wdbc.csv
    raisin.csv
    titanic.csv
outputs/
README.md
test1.py (Correction verification for Backprop Example 1)
test2.py (Correction verification for Backprop Example 2)
errorDerivative.py (Numerical Verification of Backprop 1 and 2 gradients)
```

---

## Example: Running a Model

```python
from modelSampler import ModelSampler

modelSampler = ModelSampler(filePath='datasets/wdbc.csv')
modelSampler.EPOCHS = 100
modelSampler.sampleModels(
    layerSkeleton=[[30, 16, 8, 1]],  # list of architecture you want to try
    regularization=0.01,
    stepSize=0.05,
    batchSize=32,
    stoppingCriterionCategory='epochs'
)
```

---

## Gradient and Backpropagation Verification

### Numerical Gradient Check: `errorDerivative.py`

This script numerically verifies the gradients computed by backpropagation using finite differences.

#### Usage:

```bash
python errorDerivative.py
```

You can control which example runs in `errorDerivative.py` by setting `test1` or `test2` to True or False:

- Set `test1 = True` to see results for Backprop Example 1 (`test1.py`).
- Set `test2 = True` to see results for Backprop Example 2 (`test2.py`).
- Set either to False if you want to skip that example.

```python
if __name__ == "__main__":
    test1 = True   # Set to True to run Backprop Example 1 (test1.py)
    test2 = False  # Set to True to run Backprop Example 2 (test2.py)
```

---

### Backpropagation Correctness Tests: `test1.py` and `test2.py`

These scripts verify the correctness of your forward and backward propagation implementations using provided benchmark examples.

#### Usage:
```bash
python test2.py
```