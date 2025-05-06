import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
        if self.is_binary:
            probs = self._sigmoid(X.dot(self.W) + self.b)
            return (probs > 0.5).astype(int).flatten()
        else:
            probs = self._softmax(X.dot(self.W) + self.b)
            return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        if self.is_binary:
            return self._sigmoid(X.dot(self.W) + self.b).flatten()
        else:
            return self._softmax(X.dot(self.W) + self.b)
        
    def plot_loss(self):
        plt.plot(self.train_loss_history, label='Train Loss')
        if self.test_loss_history:
            plt.plot(self.test_loss_history, label='Test Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epochs")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_roc_curve(self, y_true, y_probs):
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plt.show()


def run_on_iris():
    iris = load_iris()
    X = min_max_normalize(iris.data)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    model = LogisticRegressionScratch(learning_rate=0.5, num_iter=1000, batch_size=32, verbose=True)
    model.fit(X_train, y_train, X_test, y_test)
    preds = model.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f"Iris Accuracy: {acc:.4f}")
    model.plot_loss()

if __name__ == "__main__":
    run_on_iris()