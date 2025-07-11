import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def normalize_data(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-8)

def encode_categorical(X, y=None):
    for i in range(X.shape[1]):
        if X[:, i].dtype.kind in {'O', 'S', 'U'}:
            unique_vals = list(set(X[:, i]))
            mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
            X[:, i] = np.vectorize(mapping.get)(X[:, i])
    X = X.astype(np.float64)
    if y is not None and y.dtype.kind in {'O', 'S', 'U'}:
        unique_y = list(set(y))
        mapping_y = {val: idx for idx, val in enumerate(sorted(unique_y))}
        y = np.vectorize(mapping_y.get)(y)
        y = y.astype(np.int64)
    return X, y

def stratified_k_fold(X, y, n_splits=5, seed=42):
    np.random.seed(seed)
    unique_classes = np.unique(y)
    folds = [[] for _ in range(n_splits)]
    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        split = np.array_split(idx, n_splits)
        for i in range(n_splits):
            folds[i].extend(split[i])
    return folds

class LogisticModel:
    def __init__(self, input_dim):
        self.weights = np.zeros((input_dim, 1))
        self.bias = 0.0
        self.loss_values = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred, reg_strength):
        m = len(y_true)
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        loss += (reg_strength / (2 * m)) * np.sum(self.weights[1:] ** 2)
        return loss

    def compute_gradients(self, X, y, y_pred, reg_strength):
        m = len(y)
        dw = (X.T @ (y_pred - y)) / m
        dw[1:] += (reg_strength / m) * self.weights[1:]
        db = np.sum(y_pred - y) / m
        return dw, db

    def train(self, X, y, learning_rate=0.01, reg_strength=0.0, epochs=100, batch_size=16):
        m = len(y)
        for epoch in range(epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)
            for i in range(0, m, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx].reshape(-1, 1)

                logits = X_batch @ self.weights + self.bias
                y_pred = self.sigmoid(logits)

                dw, db = self.compute_gradients(X_batch, y_batch, y_pred, reg_strength)
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db

            logits_train = X @ self.weights + self.bias
            train_pred = self.sigmoid(logits_train)
            total_loss = self.compute_loss(y.reshape(-1, 1), train_pred, reg_strength)
            self.loss_values.append(total_loss)

    def predict(self, X):
        preds = self.sigmoid(X @ self.weights + self.bias)
        return (preds >= 0.5).astype(int).flatten()

    def calculate_metrics(self, y_pred, y_true):
        TP = TN = FP = FN = 0
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                TP += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                FN += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                FP += 1
            elif y_true[i] == 0 and y_pred[i] == 0:
                TN += 1
        acc = (TP + TN) / (TP + TN + FP + FN) * 100
        precision = TP / (TP + FP) * 100 if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) * 100 if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_values)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    data = pd.read_csv("datasets/credit_approval.csv")
    y = data['label'].values
    X = data.drop(columns=['label']).values

    X, y = encode_categorical(X, y)
    X = normalize_data(X)

    k = 5
    folds = stratified_k_fold(X, y, n_splits=k)

    batch_sizes = [16, 32, 64]
    learning_rates = [0.001, 0.01, 0.1]
    reg_lambdas = [0.001, 0.025, 0.1]

    for batch in batch_sizes:
        for lr in learning_rates:
            for reg in reg_lambdas:
                acc_list = []
                f1_list = []
                for i in range(k):
                    val_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[val_idx], y[val_idx]

                    model = LogisticModel(input_dim=X.shape[1])
                    model.train(X_train, y_train, learning_rate=lr, reg_strength=reg, epochs=200, batch_size=batch)
                    predictions = model.predict(X_test)
                    results = model.calculate_metrics(predictions, y_test)
                    acc_list.append(results['accuracy'])
                    f1_list.append(results['f1_score'])

                avg_acc = np.mean(acc_list) / 100
                avg_f1 = np.mean(f1_list) / 100
                print(f"Batch: {batch}, LR: {lr}, Reg: {reg} => Acc: {avg_acc:.4f}, F1: {avg_f1:.4f}")
