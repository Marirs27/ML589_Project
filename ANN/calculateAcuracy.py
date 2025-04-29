import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculateAccuracy(test, pred):
    correct_predictions = np.sum(np.array(test) == np.array(pred))
    total_predictions = len(test)
    accuracy = correct_predictions / total_predictions
    return round(accuracy * 100, 2)
    # return accuracy_score(test, pred) * 100
    
def confusion_matrix(test, pred, labels):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i in range(len(test)):
        actual = test[i]
        predicted = pred[i]
        cm[labels.index(actual)][labels.index(predicted)] += 1
    return cm

def calculatePrecision(test, pred, labels,pos_class_index=1):
    cm = confusion_matrix(test, pred, labels)
    tp = cm[pos_class_index, pos_class_index]
    fp = np.sum(cm[:, pos_class_index]) - tp 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return round(precision * 100, 2)
    # return precision_score(test, pred, average='binary') * 100

def calculateRecall(test, pred, labels,pos_class_index=1):
    cm = confusion_matrix(test, pred, labels)
    tp = cm[pos_class_index, pos_class_index]
    fn = np.sum(cm[pos_class_index, :]) - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return round(recall * 100, 2)
    # return recall_score(test, pred, average='binary') * 100

def calculateF1Score(test, pred, labels,pos_class_index=1):
    precision = calculatePrecision(test, pred, labels,pos_class_index)
    recall = calculateRecall(test, pred, labels,pos_class_index)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return round(f1, 2)
    # return f1_score(test, pred, average='binary') * 100


def plotConfusionMatrix(test,pred,labels,title="Confusion Matrix"):
    cm = confusion_matrix(test, pred, labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()


def plotAccuracy(accuracy):
    plt.figure(figsize=(10, 7))
    plt.bar(['Accuracy'], [accuracy])
    plt.ylabel('Score')
    plt.title('Accuracy')
    plt.show()