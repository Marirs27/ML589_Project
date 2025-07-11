# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("logreg_hyperparam_results.csv")

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# fig.suptitle("Hyperparameters vs Accuracy and F1 (Side by Side)", fontsize=16)

# # batch_size
# sns.lineplot(data=df, x="batch_size", y="accuracy", marker="o", label="Accuracy", ax=axes[0])
# sns.lineplot(data=df, x="batch_size", y="f1", marker="o", label="F1", ax=axes[0])
# axes[0].set_title("batch_size vs Accuracy & F1")
# axes[0].legend()

# # learning_rate
# sns.lineplot(data=df, x="learning_rate", y="accuracy", marker="o", label="Accuracy", ax=axes[1])
# sns.lineplot(data=df, x="learning_rate", y="f1", marker="o", label="F1", ax=axes[1])
# axes[1].set_title("learning_rate vs Accuracy & F1")
# axes[1].legend()

# # reg_lambda
# sns.lineplot(data=df, x="reg_lambda", y="accuracy", marker="o", label="Accuracy", ax=axes[2])
# sns.lineplot(data=df, x="reg_lambda", y="f1", marker="o", label="F1", ax=axes[2])
# axes[2].set_title("reg_lambda vs Accuracy & F1")
# axes[2].legend()

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

kuzushiji = fetch_openml('Kuzushiji-MNIST', version=1)
X = kuzushiji.data
y = kuzushiji.target.astype(int)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    ax = axes[i // 5, i % 5]
    img = X[y == i].iloc[0].values.reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Class {i}")
    ax.axis('off')

plt.suptitle("Example Images from Kuzushiji")
plt.tight_layout()
plt.show()

