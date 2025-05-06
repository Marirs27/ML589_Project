import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the data from the provided table
model_architectures = [
    "[6, 20, 1]",
    "[6, 8, 1]",
    "[6, 4, 8, 1]",
    "[6, 8, 4, 2, 1]",
    "[6, 4, 8, 16, 1]",
    "[6, 8, 10, 8, 1]",
    "[6, 16, 8, 2, 1]",
    "[6, 12, 7, 3, 1]"
]

accuracy = [78.804, 79.704, 79.930, 80.042, 78.576, 78.916, 81.056, 79.592]
f1_score = [71.682, 72.422, 72.210, 72.132, 71.924, 71.770, 73.140, 72.242]

# Calculate mean and std
metrics = {
    "Accuracy": accuracy,
    "F1 Score": f1_score,
}
metric_means = {key: np.mean(values) for key, values in metrics.items()}
metric_stds = {key: np.std(values) for key, values in metrics.items()}

# Set seaborn style
sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

# Plotting
plt.figure(figsize=(14, 8))

for i, (metric_name, values) in enumerate(metrics.items()):
    plt.errorbar(
        model_architectures,
        values,
        yerr=metric_stds[metric_name],
        fmt='o',
        label=f"{metric_name} (Mean: {metric_means[metric_name]:.2f})",
        color=palette[i],
        capsize=5,
        capthick=2,
        markersize=8,
        linestyle='--'
    )

# Annotate metric values
for i, (acc, f1) in enumerate(zip(accuracy, f1_score)):
    plt.annotate(f"{acc:.2f}", (i, acc), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    plt.annotate(f"{f1:.2f}", (i, f1), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

plt.xlabel('Model Architecture', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title("Neural Network Configurations Performance on Titanic Dataset", fontsize=14)
plt.xticks(ticks=range(len(model_architectures)), labels=model_architectures, rotation=45, ha='right', fontsize=10)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.show()
