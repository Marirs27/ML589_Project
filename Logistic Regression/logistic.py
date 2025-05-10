import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score
from sklearn.datasets import load_iris, fetch_openml
from sklearn import datasets
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set(style="whitegrid", palette="Set2")
palette = sns.color_palette("Set2")

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
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, num_iter=300, reg_lambda=0.01, batch_size=64, verbose=False):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.verbose = verbose
        self.train_loss_history = []
        self.test_loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _cross_entropy(self, y_true, y_pred):
        m = y_true.shape[0]
        if self.is_binary:
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(np.log(y_pred[np.arange(m), y_true]))
        reg_loss = 0.5 * self.reg_lambda * np.sum(self.W * self.W)
        return loss + reg_loss
    
    def _get_batches(self, X, y):
        indices = np.random.permutation(X.shape[0])
        for i in range(0, X.shape[0], self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            yield X[batch_idx], y[batch_idx]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if X_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)

        m, n = X_train.shape
        self.is_binary = len(np.unique(y_train)) == 2

        if self.is_binary:
            self.W = np.zeros((n, 1))
            self.b = 0
            y_train = y_train.reshape(-1, 1)
            if y_val is not None:
                y_val = y_val.reshape(-1, 1)
        else:
            k = np.max(y_train) + 1
            self.W = np.zeros((n, k))
            self.b = np.zeros((1, k))

        for epoch in range(self.num_iter):
            for X_batch, y_batch in self._get_batches(X_train, y_train):
                m_batch = X_batch.shape[0]

                if self.is_binary:
                    logits = X_batch.dot(self.W) + self.b
                    probs = self._sigmoid(logits)
                    dW = (1/m_batch) * X_batch.T.dot(probs - y_batch) + self.reg_lambda * self.W
                    db = (1/m_batch) * np.sum(probs - y_batch)
                    self.W -= self.learning_rate * dW
                    self.b -= self.learning_rate * db
                else:
                    scores = X_batch.dot(self.W) + self.b
                    probs = self._softmax(scores)
                    y_one_hot = np.zeros_like(probs)
                    y_one_hot[np.arange(m_batch), y_batch] = 1
                    dW = (1/m_batch) * X_batch.T.dot(probs - y_one_hot) + self.reg_lambda * self.W
                    db = (1/m_batch) * np.sum(probs - y_one_hot, axis=0, keepdims=True)
                    self.W -= self.learning_rate * dW
                    self.b -= self.learning_rate * db

            # Track loss
            if self.is_binary:
                probs_train = self._sigmoid(X_train.dot(self.W) + self.b)
                probs_val = self._sigmoid(X_val.dot(self.W) + self.b) if X_val is not None else None
            else:
                probs_train = self._softmax(X_train.dot(self.W) + self.b)
                probs_val = self._softmax(X_val.dot(self.W) + self.b) if X_val is not None else None

            self.train_loss_history.append(self._cross_entropy(y_train, probs_train))
            if X_val is not None:
                self.test_loss_history.append(self._cross_entropy(y_val, probs_val))

            if self.verbose and epoch % 50 == 0:
                print(f"Epoch {epoch} | Train Loss: {self.train_loss_history[-1]:.4f}")

    def predict(self, X):
        X = np.asarray(X)  # Ensure X is a numpy array
        if self.is_binary:
            probs = self._sigmoid(X.dot(self.W) + self.b)
            return (probs > 0.5).astype(int).flatten()
        else:
            probs = self._softmax(X.dot(self.W) + self.b)
            return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        X = np.asarray(X)  # Ensure X is a numpy array
        if self.is_binary:
            return self._sigmoid(X.dot(self.W) + self.b).flatten()
        else:
            return self._softmax(X.dot(self.W) + self.b)
        
    def plot_loss(self,lr, batch_size, acc, f1):
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_loss_history, label='Train Loss', color=palette[0])  # Orange
        if self.test_loss_history:
            plt.plot(self.test_loss_history, label='Test Loss', color=palette[1])  # Green
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epochs with Learning Rate: {} and Batch Size: {}".format(lr, batch_size))    
        plt.legend()
        plt.legend()
        plt.grid()
        plt.show()

    def plot_roc_curve(y_true, y_probs):
        # Sort by predicted probabilities
        desc_score_indices = np.argsort(-y_probs)
        y_true = np.array(y_true)[desc_score_indices]
        y_probs = np.array(y_probs)[desc_score_indices]
        thresholds = np.unique(y_probs)[::-1]
        tpr = []
        fpr = []
        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)
        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            TP = np.sum((y_pred == 1) & (y_true == 1))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            tpr.append(TP / P if P else 0)
            fpr.append(FP / N if N else 0)
        plt.figure(figsize=(8, 5))
        plt.plot(fpr, tpr, label="ROC Curve", color="purple")
        plt.plot([0, 1], [0, 1], '--', color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plt.show()


def hyperparam_tuning(X, y, batch_sizes, learning_rates, reg_lambdas, num_iter=300, verbose=False):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score
    results = []
    for batch_size in batch_sizes:
        for lr in learning_rates:
            for reg in reg_lambdas:
                X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=42)
                model = LogisticRegressionScratch(
                    learning_rate=lr,
                    num_iter=num_iter,
                    reg_lambda=reg,
                    batch_size=batch_size,
                    verbose=verbose
                )
                model.fit(X_train, y_train, X_val, y_val)
                preds = model.predict(X_val)
                acc = accuracy_score(y_val, preds)
                f1 = f1_score(y_val, preds, average='binary' if len(np.unique(y))==2 else 'weighted')
                results.append({
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "reg_lambda": reg,
                    "accuracy": acc,
                    "f1": f1
                })
                print(f"Batch: {batch_size}, LR: {lr}, Reg: {reg} => Acc: {acc:.4f}, F1: {f1:.4f}")
    # Convert to DataFrame for easy viewing
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv("logreg_hyperparam_results.csv", index=False)
    print("Saved hyperparameter tuning results to logreg_hyperparam_results.csv")
    return results_df

def run_on_rice(encode=True):
    import pandas as pd
    rice = pd.read_csv("datasets/rice.csv")
    y = rice['label'].values
    X = rice.drop(columns=['label']).values
    if encode:
        X, y = encode_categorical(X, y)
    X = min_max_normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=12)
    model = LogisticRegressionScratch(learning_rate=0.07, num_iter=300, batch_size=32, verbose=True)
    model.fit(X_train, y_train, X_test, y_test)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='binary')
    print(f"Rice Accuracy: {acc:.4f}", f"F1 Score: {f1:.4f}")
    model.plot_loss(lr=0.05, batch_size=32, acc=acc, f1=f1)
    model.plot_roc_curve(y_test, model.predict_proba(X_test))

def run_on_parkinsons(encode=True):
    import pandas as pd
    parkinsons = pd.read_csv("datasets/parkinsons.csv")
    y = parkinsons['Diagnosis'].values
    X = parkinsons.drop(columns=['Diagnosis']).values
    if encode:
        X, y = encode_categorical(X, y)
    X = min_max_normalize(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=12)
    # model = LogisticRegressionScratch(learning_rate=0.5, num_iter=1000, batch_size=52, verbose=True)
    # model.fit(X_train, y_train, X_test, y_test)
    # preds = model.predict(X_test)
    # acc = accuracy_score(y_test, preds)
    # f1 = f1_score(y_test, preds, average='binary')
    # print(f"Parkinsons Accuracy: {acc:.4f}", f"F1 Score: {f1:.4f}")
    # model.plot_loss(lr=0.05, batch_size=32, acc=acc, f1=f1)
    # model.plot_roc_curve(y_test, model.predict_proba(X_test))
    batch_sizes = [16, 32, 64]
    learning_rates = [0.001, 0.01, 0.1]
    reg_lambdas = [0.001, 0.025, 0.1]
    hyperparam_tuning(X, y, batch_sizes, learning_rates, reg_lambdas)

def run_on_credit(encode=True):
    import pandas as pd
    credit_approval = pd.read_csv("datasets/credit_approval.csv")
    y = credit_approval['label'].values
    X = credit_approval.drop(columns=['label']).values
    if encode:
        X, y = encode_categorical(X, y)
    X = min_max_normalize(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=12)
    # model = LogisticRegressionScratch(learning_rate=0.5, num_iter=1000, batch_size=52, verbose=True)
    # model.fit(X_train, y_train, X_test, y_test)
    # preds = model.predict(X_test)
    # acc = accuracy_score(y_test, preds)
    # f1 = f1_score(y_test, preds, average='binary')
    # print(f"Credit Approval Accuracy: {acc:.4f}", f"F1 Score: {f1:.4f}")
    # model.plot_loss(lr=0.01, batch_size=64, acc=acc, f1=f1)
    # LogisticRegressionScratch.plot_roc_curve(y_test, model.predict_proba(X_test))
    # After loading and preprocessing your data (X, y):
    batch_sizes = [10, 16, 32, 64]
    learning_rates = [0.01, 0.05, 0.1]
    reg_lambdas = [0.001, 0.01, 0.1]
    hyperparam_tuning(X, y, batch_sizes, learning_rates, reg_lambdas)

def run_digits_mnist():
    digits = datasets.load_digits(return_X_y=True)
    X = min_max_normalize(digits[0])
    y = digits[1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    # model = LogisticRegressionScratch(learning_rate=0.05, num_iter=1000, batch_size=32, verbose=True)
    # model.fit(X_train, y_train, X_test, y_test)
    # preds = model.predict(X_test)
    # acc = np.mean(preds == y_test)
    # f1 = f1_score(y_test, preds, average='weighted')
    # print(f"Digits Accuracy: {acc:.4f}", f"F1 Score: {f1:.4f}")
    # model.plot_loss(lr=0.01, batch_size=64,acc=acc, f1=f1)
    batch_sizes = [10, 16, 32, 64]
    learning_rates = [0.01, 0.05, 0.1]
    reg_lambdas = [0.001, 0.01, 0.1]
    hyperparam_tuning(X, y, batch_sizes, learning_rates, reg_lambdas)

def run_kuzushiji_mnist():
    kuzushiji = fetch_openml('Kuzushiji-MNIST', version=1)
    X = kuzushiji.data.astype(float)
    y = kuzushiji.target.astype(int)
    X = min_max_normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    # model = LogisticRegressionScratch(
    #     learning_rate=0.01, 
    #     num_iter=500, 
    #     batch_size=28, 
    #     reg_lambda=0.01, 
    #     verbose=True
    # )
    # model.fit(X_train, y_train, X_test, y_test)
    # preds = model.predict(X_test)
    # acc = np.mean(preds == y_test)
    # f1 = f1_score(y_test, preds, average='weighted')
    # print(f"Kuzushiji-MNIST Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    # model.plot_loss(lr=0.01, batch_size=64, acc=acc, f1=f1)
    
    batch_sizes = [10, 16, 32, 64]
    learning_rates = [0.01, 0.05, 0.1]
    reg_lambdas = [0.001, 0.01, 0.1]
    hyperparam_tuning(X, y, batch_sizes, learning_rates, reg_lambdas)




if __name__ == "__main__":
    # run_kuzushiji_mnist()
    # run_digits_mnist()
    # run_on_parkinsons()
    run_on_credit()