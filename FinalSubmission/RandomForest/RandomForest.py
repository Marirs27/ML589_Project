import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets

LABEL_NAME = 'Diagnosis' 

def get_bootstrap_sample(data):
    return data.sample(n=len(data), replace=True)

def _get_entropy(y):
    d = len(y)
    if d == 0:
        return 0
    _, y_mapped = np.unique(y, return_inverse=True)
    counts = np.bincount(y_mapped)
    probs = counts / d
    return -np.sum(probs * np.log2(probs))

def _get_info_gain(data, split_attribute):
    label_col = LABEL_NAME  # <-- Fixed to use last column as label
    dataset_entropy = _get_entropy(data[label_col])
    if pd.api.types.is_numeric_dtype(data[split_attribute]):
        return _get_numeric_info_gain(data, split_attribute, label_col, dataset_entropy)
    else:
        info_gain = dataset_entropy
        splits = data[split_attribute].unique()
        for split in splits:
            split_subset = data[data[split_attribute] == split]
            info_gain -= len(split_subset) / len(data) * _get_entropy(split_subset[label_col])
        return info_gain, None

def _get_numeric_info_gain(data, split_attribute, label_col, dataset_entropy):
    sorted_thresh = np.sort(data[split_attribute].unique())
    if len(sorted_thresh) < 2:
        return 0, None
    splits = (sorted_thresh[:-1] + sorted_thresh[1:]) / 2
    best_gain, threshold = -float('inf'), None
    for split in splits:
        left = data[data[split_attribute] <= split]
        right = data[data[split_attribute] > split]
        if left.empty or right.empty:
            continue
        gain = dataset_entropy - (
            (len(left) / len(data)) * _get_entropy(left[label_col]) +
            (len(right) / len(data)) * _get_entropy(right[label_col])
        )
        if gain > best_gain:
            best_gain = gain
            threshold = split
    return best_gain, threshold

def create_decision_tree(data, attributes, M, max_depth=None, min_sample_split=2, min_info_gain=0.0, depth=0):
    label_col = LABEL_NAME

    # Base Case 1: All labels are the same
    if data[label_col].nunique() == 1:
        return data[label_col].iloc[0]

    # Base Case 2: No attributes left to split
    if len(attributes) == 0:  # Fixed condition
        return data[label_col].mode()[0]

    # Base Case 3: Maximum depth reached
    if max_depth is not None and depth >= max_depth:
        return data[label_col].mode()[0]

    # Base Case 4: Not enough samples to split
    if len(data) < min_sample_split:
        return data[label_col].mode()[0]

    # Base Case 5: Entropy is below the minimum threshold
    current_entropy = _get_entropy(data[label_col])
    if current_entropy <= min_info_gain:
        return data[label_col].mode()[0]

    # Select sqrt(M) features
    m = min(M, len(attributes))
    chosen_attributes = np.random.choice(attributes, m, replace=False)

    # Find the best attribute and threshold to split
    best_gain, best_split, best_threshold = -float('inf'), None, None
    for attr in chosen_attributes:
        gain, threshold = _get_info_gain(data, attr)
        if gain > best_gain:
            best_gain = gain
            best_split = attr
            best_threshold = threshold

    # If no valid split is found, return the majority class
    if best_gain == 0 or best_split is None:
        return data[label_col].mode()[0]

    # Split the data and create branches
    branches = {}
    new_attributes = [attr for attr in attributes if attr != best_split]

    if best_threshold is not None:  # Numerical split
        left = data[data[best_split] <= best_threshold]
        right = data[data[best_split] > best_threshold]

        # Base Case 6: No valid split (one branch is empty)
        if left.empty or right.empty:
            return data[label_col].mode()[0]

        branches[("le", best_threshold)] = create_decision_tree(
            left, new_attributes, M, max_depth, min_sample_split, min_info_gain, depth + 1
        )
        branches[("gt", best_threshold)] = create_decision_tree(
            right, new_attributes, M, max_depth, min_sample_split, min_info_gain, depth + 1
        )
    else:  # Categorical split
        for val in data[best_split].unique():
            subset = data[data[best_split] == val]
            branches[val] = create_decision_tree(
                subset, new_attributes, M, max_depth, min_sample_split, min_info_gain, depth + 1
            )

    return (best_split, branches, data[label_col].mode()[0], best_threshold)

def cross_validation_stratified(data, k=5, seed=19):
    np.random.seed(seed)
    data = data.sample(frac=1).reset_index(drop=True)
    label_col = LABEL_NAME
    target = data[label_col]
    folds = [[] for _ in range(k)]
    for label in target.unique():
        idx = list(data[target == label].index)
        np.random.shuffle(idx)
        for i, index in enumerate(idx):
            folds[i % k].append(index)
    return [(data.iloc[sum(folds[:i] + folds[i+1:], [])].reset_index(drop=True),
             data.iloc[folds[i]].reset_index(drop=True)) for i in range(k)]

def tree_prediction(tree, row):
    while isinstance(tree, tuple):
        attr, branches, default, threshold = tree
        val = row[attr]
        if threshold is not None:
            tree = branches.get(("le", threshold) if val <= threshold else ("gt", threshold), default)
        else:
            tree = branches.get(val, default)
    return tree

def fit_random_forest(data, attributes, ntree, min_samples_split=2, max_depth=15, min_info_gain=0.01):
    M = max(1, int(np.sqrt(len(attributes))))
    return [create_decision_tree(get_bootstrap_sample(data), attributes, M, max_depth, min_samples_split, min_info_gain) for _ in range(ntree)]

def predict_random_forest(forest, row):
    votes = [tree_prediction(tree, row) for tree in forest]
    return pd.Series(votes).mode()[0]

def calculate_metrics(y_true, y_pred):
    tp, fp, fn, tn = 0, 0, 0, 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            if true == 1:
                tp += 1
            else:
                tn += 1
        else:
            if true == 1:
                fn += 1
            else:
                fp += 1
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return accuracy, precision, recall, f1


def plot_color_metrics(results, dataset_name):
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = ['#fedd51', '#8ab25e', '#ef78a8', '#f3a6a6', '#bb7641']
    markers = ["o", "s", "D", "^"]  # Markers for each metric

    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        means = results[metric]
        stds = results.get(f"{metric}_std", [0] * len(means))  # Default std to 0 if not provided

        # Plot with error bars and connecting lines
        plt.errorbar(
            results["nTree"],  # Corrected
            means,
            yerr=stds,
            fmt=f"{markers[i]}-",  # Marker with a line
            color=colors[i],
            ecolor="gray",
            elinewidth=1.5,
            capsize=4,
            label=metric.capitalize(),
            linewidth=2,  # Line width for better visibility
        )

        # Annotate each point with mean ± std
        for x, y, std in zip(results["nTree"], means, stds):
            plt.annotate(
                f"{y:.4f}±{std:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

        # Set titles, labels, and grid
        plt.title(f"{metric.capitalize()} vs Number of Trees", fontsize=12)
        plt.xlabel("Number of Trees", fontsize=10)
        plt.ylabel(metric.capitalize(), fontsize=10)
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout and add a main title
    plt.tight_layout()
    plt.suptitle(f"Random Forest Performance on {dataset_name}", fontsize=16, y=1.02)
    plt.savefig(f"{dataset_name}_random_forest_metrics_color.png")
    plt.show()


def plot_all_metrics_color(results, dataset_name):
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = ['#fedd51', '#8ab25e', '#ef78a8', '#f3a6a6', '#bb7641']
    markers = ["o", "s", "D", "^"]  # Markers for each metric

    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        plt.plot(
            results["nTree"], 
            results[metric], 
            marker=markers[i], 
            color=colors[i], 
            label=metric.capitalize()
        )
    plt.title(f"Random Forest Performance on {dataset_name}")
    plt.xlabel("Number of Trees")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dataset_name}_random_forest_all_metrics_color.png")
    plt.show()


def hyperparameter_tuning(data, attributes):
    import pandas as pd
    min_samples = [2, 5, 10]
    max_depths = [10, 15, 30]
    min_info_gains = [0.01, 0.05, 0.1]
    n_tree = [20]
    best_params = None
    best_f1 = 0
    results = []  # To store results for DataFrame

    for samples in min_samples:
        for depth in max_depths:
            for gain in min_info_gains:
                for ntree in n_tree:
                    accs, f1s = [], []
                    for train_df, test_df in cross_validation_stratified(data, k=10):
                        forest = fit_random_forest(train_df, attributes, ntree, min_samples_split=samples, max_depth=depth, min_info_gain=gain)
                        y_true = test_df[LABEL_NAME].tolist()
                        y_pred = [predict_random_forest(forest, row) for _, row in test_df.iterrows()]
                        acc, _, _, f1 = calculate_metrics(y_true, y_pred)
                        f1s.append(f1)
                        accs.append(acc)

                    avg_f1 = np.mean(f1s)
                    avg_acc = np.mean(accs)
                    print(f"Samples: {samples}, Depth: {depth}, Gain: {gain}, F1: {avg_f1:.4f}, Acc: {avg_acc:.4f}")
                    results.append({
                        "min_samples": samples,
                        "max_depth": depth,
                        "min_info_gain": gain,
                        "n_tree": ntree,
                        "avg_f1": avg_f1,
                        "avg_acc": avg_acc
                    })
                    if avg_f1 > best_f1:
                        best_f1 = avg_f1
                        best_params = (samples, depth, gain)
    # Save results to DataFrame and CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("rf_hyperparam_results.csv", index=False)
    print("Saved hyperparameter tuning results to rf_hyperparam_results.csv")
    return best_params

def run_random_forest(filename, nTrees=[5, 10, 20, 30, 40, 50], k=5, min_samples_split= 10, max_depth= 10, min_info_gain= 0.1):
    print(f"Running Random Forest on {filename}")
    data = pd.read_csv(filename)
    label_col = LABEL_NAME 

    # --- Encode categorical features (excluding label) ---
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.drop(label_col, errors='ignore')
    if len(categorical_features) > 0:
        data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    
    # --- Encode label column if it's categorical ---
    if data[label_col].dtype == 'object' or data[label_col].dtype.name == 'category':
        le = LabelEncoder()
        data[label_col] = le.fit_transform(data[label_col])

    results = {
        'nTree': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy_std': [],
        'precision_std': [],
        'recall_std': [],
        'f1_std': []
    }
    for ntree in nTrees:
        accs, precs, recs, f1s = [], [], [], []
        for train_df, test_df in cross_validation_stratified(data, k):
            features = [col for col in train_df.columns if col != LABEL_NAME]
            target = LABEL_NAME
            forest = fit_random_forest(train_df, attributes=features, ntree=ntree, min_samples_split=min_samples_split, max_depth=max_depth, min_info_gain=min_info_gain)
            y_true = test_df[target].tolist()
            y_pred = [predict_random_forest(forest, row) for _, row in test_df.iterrows()]
            acc, prec, rec, f1 = calculate_metrics(y_true, y_pred)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
        results['nTree'].append(ntree)
        results['accuracy'].append(np.mean(accs))
        results['precision'].append(np.mean(precs))
        results['recall'].append(np.mean(recs))
        results['f1'].append(np.mean(f1s))
        results['accuracy_std'].append(np.std(accs))
        results['precision_std'].append(np.std(precs))
        results['recall_std'].append(np.std(recs))
        results['f1_std'].append(np.std(f1s))
        print(f"Trees: {ntree}, Accuracy: {np.mean(accs):.4f}, Precision: {np.mean(precs):.4f}, Recall: {np.mean(recs):.4f}, F1: {np.mean(f1s):.4f}")
    plot_color_metrics(results, dataset_name=filename.split("/")[-1].replace(".csv", ""))
    plot_all_metrics_color(results, dataset_name=filename.split("/")[-1].replace(".csv", ""))

if __name__ == "__main__":
    # run_random_forest("datasets/credit_approval.csv")  
    run_random_forest("datasets/parkinsons.csv")  
    # digits = datasets.load_digits(return_X_y=True)
    # digits_dataset_X = digits[0]
    # digits_dataset_y = digits[1]
    # data = pd.DataFrame(digits_dataset_X, columns=[f'pixel_{i}' for i in range(digits_dataset_X.shape[1])])
    # data['label'] = digits_dataset_y
    # # Save to CSV
    # data.to_csv("digits.csv", index=False)

    # Now run your random forest on this file
    # run_random_forest("digits.csv")
    # run_random_forest("datasets/rice.csv")


    # Hyperparameter tuning
    # data = pd.read_csv("datasets/parkinsons.csv")
    # # --- Encode categorical features (excluding label) ---
    # categorical_features = data.select_dtypes(include=['object', 'category']).columns.drop(LABEL_NAME, errors='ignore')
    # if len(categorical_features) > 0:
    #     data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

    # # --- Encode label column if it's categorical ---
    # if data[LABEL_NAME].dtype == 'object' or data[LABEL_NAME].dtype.name == 'category':
    #     le = LabelEncoder()
    #     data[LABEL_NAME] = le.fit_transform(data[LABEL_NAME])

    # # Perform hyperparameter tuning
    # attributes = [col for col in data.columns if col != LABEL_NAME]
    # best_params = hyperparameter_tuning(data, attributes)
    # print(f"Best Parameters: {best_params}")
    


