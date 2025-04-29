'''
Train a neural network and evaluate it using the stratified cross-validation technique discussed in
class. You should train neural networks using different values for the regularization parameter, λ,
and using different architectures. You can decide which architectures you will evaluate. As an
example, consider testing networks with 1, 2, 3, and 4 hidden layers, and with various numbers of
neurons per layer: e.g., 2, 4, 8, 16.
2. For each trained neural network (i.e., for each combination of architecture and regularization that
you tested), you should measure the resulting model’s accuracy and F1 score.
5
3. Based on these experiments, you should create, for each dataset and for each of the metrics described
above, a table summarizing the corresponding results (i.e., you should show the value of each
performance metric for each type of neural network that you tested, on both datasets). You should
test at least 6 neural network architectures on each dataset
'''
import numpy as np
from layer import Layer
from forwardPropagation import ForwardPropagation
from backPropagation import BackPropagation
from dataProcess import DataPreprocessor
from trainModel import TrainModel
import matplotlib.pyplot as plt
import seaborn as sns


WDBC_LAYERS_SKELETON = [
    # [12, 10, 8, 10, 1],  # 4 hidden layers
    [10, 32, 1],         # 2 hidden layers
    # [18, 1],             # 1 hidden layer
    # [15, 12, 8, 12 ,10,  1],         # 2 hidden layers
]
LOAN_LAYERS_SKELETON = [[5,1], [12,1],[5, 10, 1], [10,5,8,1]]
TITANIC_LAYERS_SKELETON = [[20,1], [8,1],[16,4,8,1],[8,4,2,1]]
RAISIN_LAYERS_SKELETON =  [[1], [5,1],[10, 8, 1],[10, 16, 8, 4, 1],[10, 8, 6, 8, 1]]
class ModelSampler:
    EPSILON = 0.01
    REGULARIZATION_VALUES = [0.01,0.025]
    STEP_SIZE_VALUES = [0.01, 0.05]
    BATCH_SIZE_VALUES = [5, 10]
    K_FOLD = 5
    EPOCHS = 100
    
    def __init__(self, filePath, label='label', seed=42):
        self.filePath = filePath
        self.preprocessor = DataPreprocessor(filePath=filePath, kFold=self.K_FOLD, labelColumn=label, randomSeed=seed)
        self.preprocessor.load_data()
        self.preprocessor.encodeCategorical()
        self.preprocessor.normalizeData()
        self.preprocessor.stratifiedKFold()
        self.preprocessor.printDataDetails()
        self.trainModels = []
        self.accuracy = []
        self.f1Score = []
        self.confusionMatrix = []
        self.precision = []
        self.recall = []

    def sampleModels(self, layerSkeleton, regularization=0.01, stepSize=0.01, batchSize=10, thresholdValue=0.5, stoppingCriterionCategory='epochs'):
        # Store all the models and their metrics to plot
        accuracy = []
        f1Score = []
        precision = []
        recall = []
        models = []

        for layers in layerSkeleton:
            # Add input layer size to the beginning of the architecture
            l = layers.copy()
            l.insert(0, self.preprocessor.data.shape[1] - 1)

            # Create and train the model
            model = TrainModel(self.preprocessor, l, self.EPSILON, batchSize, regularization=regularization, stepSize=stepSize, threshold=thresholdValue, epoch=self.EPOCHS)
            print("\n\n")
            print(f"Model with layers {l}, regularization {regularization}, batch size {batchSize}, step size {stepSize} created successfully")

            # Train the model using k-fold cross-validation and get Learbing curve (mtric vs epoch list)
            accLC, preLC, recLC, f1LC = model.kFoldTrainTest(stoppingCriterion=stoppingCriterionCategory)

            # Append metrics and model to their respective lists
            accuracy.append(accLC)
            f1Score.append(f1LC)
            precision.append(preLC)
            recall.append(recLC)
            models.append(model)

            plotLearningCurve(accLC, f1LC, preLC, recLC, model, title="Model Performance of {} with architecture {} regularization={}, stepSize={}, batchSize={}".format(self.filePath.split('/')[1],l,regularization, stepSize, batchSize))

        # Plot the metrics
        # plotMetrics(accuracy, f1Score, precision, recall, models, title="Model Performance of {} with regularization={}, stepSize={}, batchSize={}".format(self.filePath.split('/')[2],regularization, stepSize, batchSize))
        print("Model sampling completed successfully")

def plotLearningCurve(accuracy, f1Score, precision, recall, model:TrainModel, title="Model Performance"):
    # Set Seaborn style and color palette
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2")

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Generate epoch numbers for the x-axis
    epochs = list(range(1, len(accuracy) + 1))

    # Ensure the order of metric names matches the order of metric values
    metric_names = ["Accuracy", "F1 Score", "Recall", "Precision"]  # Corrected order
    metric_values = [accuracy, f1Score, recall, precision]  # Corrected order

    # Plot each metric
    for i, (metric_name, values) in enumerate(zip(metric_names, metric_values)):
        plt.plot(
            epochs,
            values,
            # marker='o',
            label=metric_name,
            color=palette[i],
            linestyle='-'
        )

    # Customize the plot
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.show()

def plotMetrics(accuracy, f1Score, precision, recall, models, title="Model Performance"):
    # Convert models to string representations of their architectures
    model_architectures = [f"Layers: {model.layersSkeleton}" for model in models]

    # Calculate mean and standard deviation for each metric
    metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1Score,
        "Precision": precision,
        "Recall": recall
    }
    metric_means = {key: np.mean(values) for key, values in metrics.items()}
    metric_stds = {key: np.std(values) for key, values in metrics.items()}

    # Set Seaborn style and color palette
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2")

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Plot each metric with error bars (mean ± std deviation)
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

    # Annotate metric values on every point
    for i, (acc, f1, pre, rec) in enumerate(zip(accuracy, f1Score, precision, recall)):
        plt.annotate(f"{acc:.2f}", (i, accuracy[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        plt.annotate(f"{f1:.2f}", (i, f1Score[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        plt.annotate(f"{pre:.2f}", (i, precision[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        plt.annotate(f"{rec:.2f}", (i, recall[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(title, fontsize=14)
    # plt.ylim(0,100)
    plt.xticks(ticks=range(len(model_architectures)), labels=model_architectures, rotation=45, ha='right', fontsize=10)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    modelSampler = ModelSampler(filePath='datasets/credit_approval.csv',label='label', seed=42)
    modelSampler.sampleModels(layerSkeleton = WDBC_LAYERS_SKELETON ,regularization=0.01, stepSize=0.01, batchSize=32,stoppingCriterionCategory='epochs')
    print("Model sampling completed successfully")


