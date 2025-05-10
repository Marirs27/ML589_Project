# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, f1_score, make_scorer
# import numpy as np
# import matplotlib.pyplot as plt

# # Load credit dataset
# credit = pd.read_csv("datasets/parkinsons.csv")
# label_col = 'Diagnosis'  # Change if needed

# # Encode categorical features
# for col in credit.select_dtypes(include=['object', 'category']).columns:
#     if col != label_col:
#         credit[col] = LabelEncoder().fit_transform(credit[col])
# if credit[label_col].dtype == 'object' or str(credit[label_col].dtype).startswith('category'):
#     credit[label_col] = LabelEncoder().fit_transform(credit[label_col])

# # Prepare X, y and normalize
# X = credit.drop(columns=[label_col]).values
# y = credit[label_col].values
# X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)

# k_values = range(5, 40, 2)
# acc_scores = []
# f1_scores = []
# acc_stds = []
# f1_stds = []

# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     acc = []
#     f1 = []
#     for train_idx, test_idx in cv.split(X, y):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
#         knn.fit(X_train, y_train)
#         y_pred = knn.predict(X_test)
#         acc.append(accuracy_score(y_test, y_pred) * 100)
#         f1.append(f1_score(y_test, y_pred, average='binary') * 100)
#     acc_scores.append(np.mean(acc))
#     f1_scores.append(np.mean(f1))
#     acc_stds.append(np.std(acc))
#     f1_stds.append(np.std(f1))

# colors = ('skyblue','blue')
# plt.figure(figsize=(12,6))
# plt.errorbar(k_values, acc_scores, yerr=acc_stds, fmt='o', capsize=5,
#              color=colors[0], ecolor=colors[1], elinewidth=0.8, capthick=1, markersize=6, label='Accuracy')
# plt.plot(k_values, acc_scores, linestyle='-', marker='o', color=colors[0])
# plt.errorbar(k_values, f1_scores, yerr=f1_stds, fmt='o', capsize=5,
#              color='salmon', ecolor='red', elinewidth=0.8, capthick=1, markersize=6, label='F1 Score')
# plt.plot(k_values, f1_scores, linestyle='-', marker='o', color='salmon')
# for i, (k, acc, f1) in enumerate(zip(k_values, acc_scores, f1_scores)):
#     plt.annotate(f'{acc:.2f}', (k, acc), textcoords="offset points", xytext=(0,10), ha='center', color=colors[1])
#     plt.annotate(f'{f1:.2f}', (k, f1), textcoords="offset points", xytext=(0,-15), ha='center', color='red')
# plt.xlabel('K value')
# plt.ylabel('Score (%)')
# plt.grid()
# plt.legend()
# plt.title('Accuracy and F1 Score vs K value')
# plt.show()

# best_k_acc = k_values[np.argmax(acc_scores)]
# best_acc = max(acc_scores)
# best_k_f1 = k_values[np.argmax(f1_scores)]
# best_f1 = max(f1_scores)

# print(f"Best Accuracy: k={best_k_acc}, score={best_acc:.2f}")
# print(f"Best F1 Score: k={best_k_f1}, score={best_f1:.2f}")



from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# Load KMNIST dataset
kuzushiji = fetch_openml('Kuzushiji-MNIST', version=1, as_frame=False)
X = kuzushiji.data / 255.0  # Normalize pixel values to [0, 1]
y = kuzushiji.target.astype(int)

# For speed, use a subset (e.g., 10000 samples)
X = X[:10000]
y = y[:10000]

# Single train/test split (no k-fold)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k_values = range(5, 40, 2)
acc_scores = []
f1_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    acc_scores.append(acc)
    f1_scores.append(f1)

colors = ('skyblue','blue')
plt.figure(figsize=(12,6))
plt.plot(k_values, acc_scores, linestyle='-', marker='o', color=colors[0], label='Accuracy')
plt.plot(k_values, f1_scores, linestyle='-', marker='o', color='salmon', label='F1 Score')
for i, (k, acc, f1) in enumerate(zip(k_values, acc_scores, f1_scores)):
    plt.annotate(f'{acc:.2f}', (k, acc), textcoords="offset points", xytext=(0,10), ha='center', color=colors[1])
    plt.annotate(f'{f1:.2f}', (k, f1), textcoords="offset points", xytext=(0,-15), ha='center', color='red')
plt.xlabel('K value')
plt.ylabel('Score (%)')
plt.grid()
plt.legend()
plt.title('Accuracy and F1 Score vs K value')
plt.show()

best_k_acc = k_values[np.argmax(acc_scores)]
best_acc = max(acc_scores)
best_k_f1 = k_values[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

print(f"Best Accuracy: k={best_k_acc}, score={best_acc:.2f}")
print(f"Best F1 Score: k={best_k_f1}, score={best_f1:.2f}")