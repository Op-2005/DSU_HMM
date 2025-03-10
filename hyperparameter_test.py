import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import ParameterGrid

from data_processor import FinancialDataLoader, discretize_data
from hmm_model import HiddenMarkovModel

np.random.seed(42)
torch.manual_seed(42)

device = 'cpu'
print(f"Using device: {device}")


def run_hyperparameter_test(
    data_file='financial_data.csv',
    feature='sp500 high-low',
    target='sp500 close',
    mode='classification',
    states=3,
    observations=15,
    steps=20,
    discr_strategy='equal_freq',
    class_threshold=0.5,
    use_feature_for_hmm=False,
    direct_states=False,
    test_size=0.2,
    val_size=0.2,
    use_validation=True,
    final_test=False
):
    """Run a single hyperparameter test with the given parameters"""

    print("\n" + "="*70)
    print(
        f"RUNNING TEST: states={states}, observations={observations}, strategy={discr_strategy}")
    print(
        f"          steps={steps}, direct_states={direct_states}, feature={feature}")
    print("="*70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, data_file)
    normalize = True

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data_loader = FinancialDataLoader(
        file_path=file_path,
        target_column=target,
        features=[feature],
        normalize=normalize
    )

    log_returns_col = data_loader.add_log_returns(target)
    label_col = data_loader.add_regime_labels(
        log_returns_col, threshold=0.0, window=5)

    # Implement train/validation/test split
    total_samples = len(data_loader.data)
    test_size_actual = int(total_samples * test_size)
    train_val_size = total_samples - test_size_actual

    indices = np.arange(total_samples)
    train_val_indices = indices[:train_val_size]
    test_indices = indices[train_val_size:]

    # Create test dataset
    test_data = data_loader.data.iloc[test_indices].copy()
    test_loader = FinancialDataLoader(
        file_path=None, target_column=target, features=[feature],
        normalize=normalize, data=test_data, device=device
    )

    if final_test:
        # Use all non-test data for training
        train_data = data_loader.data.iloc[train_val_indices].copy()
        train_loader = FinancialDataLoader(
            file_path=None, target_column=target, features=[feature],
            normalize=normalize, data=train_data, device=device
        )

        # Use test data for evaluation
        X_train = train_loader.data[feature].values
        X_eval = test_loader.data[feature].values
        y_train = train_loader.data[log_returns_col].values
        y_eval = test_loader.data[log_returns_col].values
        train_labels = train_loader.data[label_col].values
        eval_labels = test_loader.data[label_col].values
    else:
        # Split train/val from the train_val set
        val_size_actual = int(train_val_size * val_size)
        train_size = train_val_size - val_size_actual

        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]

        # Create datasets
        train_data = data_loader.data.iloc[train_indices].copy()
        val_data = data_loader.data.iloc[val_indices].copy()

        train_loader = FinancialDataLoader(
            file_path=None, target_column=target, features=[feature],
            normalize=normalize, data=train_data, device=device
        )

        val_loader = FinancialDataLoader(
            file_path=None, target_column=target, features=[feature],
            normalize=normalize, data=val_data, device=device
        )

        # Use validation set for evaluation
        X_train = train_loader.data[feature].values
        X_eval = val_loader.data[feature].values
        y_train = train_loader.data[log_returns_col].values
        y_eval = val_loader.data[log_returns_col].values
        train_labels = train_loader.data[label_col].values
        eval_labels = val_loader.data[label_col].values

    # Data for HMM
    if use_feature_for_hmm:
        hmm_train_data = X_train
        hmm_eval_data = X_eval
    else:
        hmm_train_data = y_train
        hmm_eval_data = y_eval

    X_train_discrete = discretize_data(
        hmm_train_data, num_bins=observations, strategy=discr_strategy)
    X_eval_discrete = discretize_data(
        hmm_eval_data, num_bins=observations, strategy=discr_strategy)

    # Train the HMM model
    T, E, T0 = initialize_hmm_params(states, observations)
    hmm = HiddenMarkovModel(T, E, T0, device='cpu', maxStep=steps)

    start_time = time.time()
    T0, T, E, converged = hmm.Baum_Welch_EM(X_train_discrete)
    training_time = time.time() - start_time

    # Evaluate the model
    eval_metrics = hmm.evaluate(
        X_eval_discrete,
        mode=mode,
        actual_values=hmm_eval_data,
        actual_labels=eval_labels,
        observation_map=None,
        class_threshold=class_threshold,
        direct_states=direct_states
    )

    # Add test parameters to the metrics
    eval_metrics.update({
        'states': states,
        'observations': observations,
        'steps': steps,
        'discr_strategy': discr_strategy,
        'direct_states': direct_states,
        'feature': feature,
        'training_time': training_time,
        'converged': converged
    })

    # Remove large arrays that we don't need for the report
    if 'states_seq' in eval_metrics:
        del eval_metrics['states_seq']
    if 'predicted_labels' in eval_metrics:
        del eval_metrics['predicted_labels']

    return eval_metrics


def initialize_hmm_params(num_states, num_observations):
    """Initialize HMM parameters (copied from train.py)"""
    T = np.ones((num_states, num_states)) / num_states
    T = T + np.random.uniform(0, 0.1, T.shape)
    T = T / T.sum(axis=1, keepdims=True)

    E = np.ones((num_observations, num_states)) / num_states
    E = E + np.random.uniform(0, 0.1, E.shape)
    E = E / E.sum(axis=1, keepdims=True)

    T0 = np.ones(num_states) / num_states

    return T, E, T0


def run_hyperparameter_grid_search():
    """Run a grid search over hyperparameters"""

    # Define the hyperparameter grid
    param_grid = {
        'states': [2, 3, 4, 5, 6, 8],
        'observations': [10, 20, 30, 40],
        'discr_strategy': ['equal_freq', 'equal_width', 'kmeans'],
        'direct_states': [True, False],
        'feature': ['sp500 high-low', 'sp500 close'],
        'steps': [30]  # Keep fixed to save time
    }

    # Calculate total combinations
    grid = list(ParameterGrid(param_grid))
    print(f"Running grid search with {len(grid)} parameter combinations")

    # Since we need to run at least 12 tests, let's select a subset of meaningful combinations
    selected_params = [
        {'states': 2, 'observations': 10, 'discr_strategy': 'equal_freq',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 2, 'observations': 10, 'discr_strategy': 'kmeans',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 3, 'observations': 20, 'discr_strategy': 'equal_freq',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 3, 'observations': 20, 'discr_strategy': 'kmeans',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 4, 'observations': 30, 'discr_strategy': 'equal_freq',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 4, 'observations': 30, 'discr_strategy': 'kmeans',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 5, 'observations': 40, 'discr_strategy': 'equal_freq',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 5, 'observations': 40, 'discr_strategy': 'kmeans',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 6, 'observations': 30, 'discr_strategy': 'equal_freq',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 6, 'observations': 30, 'discr_strategy': 'kmeans',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 6, 'observations': 40, 'discr_strategy': 'equal_freq',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 6, 'observations': 40, 'discr_strategy': 'kmeans',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 6, 'observations': 40, 'discr_strategy': 'kmeans',
            'direct_states': True, 'feature': 'sp500 close'},
        {'states': 8, 'observations': 40, 'discr_strategy': 'kmeans',
            'direct_states': True, 'feature': 'sp500 high-low'},
        {'states': 8, 'observations': 40, 'discr_strategy': 'kmeans',
            'direct_states': False, 'feature': 'sp500 high-low'}
    ]

    results = []

    for params in tqdm(selected_params):
        try:
            print(f"Running test with parameters: {params}")

            result = run_hyperparameter_test(
                states=params['states'],
                observations=params['observations'],
                discr_strategy=params['discr_strategy'],
                direct_states=params['direct_states'],
                feature=params['feature'],
                steps=30  # Fixed steps for all tests
            )

            # Add the result to our list
            results.append(result)

            # Save progress after each test
            with open('hyperparameter_results.json', 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")

    return results


def generate_report(results):
    """Generate a comprehensive report from the test results"""

    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Create the report markdown
    report = "# HMM Hyperparameter Optimization Report\n\n"

    # Add overview
    report += "## Overview\n\n"
    report += f"This report presents the results of hyperparameter optimization for an HMM (Hidden Markov Model) used for financial market regime classification. "
    report += f"We tested {len(results)} different hyperparameter combinations to find the optimal configuration. "
    report += f"The main metrics evaluated were accuracy, precision, recall, and F1-score.\n\n"

    # Add best results by different metrics
    report += "## Best Results\n\n"

    best_accuracy = df.loc[df['accuracy'].idxmax()]
    best_f1 = df.loc[df['f1_score'].idxmax()]

    report += "### Best by Accuracy\n\n"
    report += f"- **Accuracy**: {best_accuracy['accuracy']:.4f}\n"
    report += f"- **Precision**: {best_accuracy['precision']:.4f}\n"
    report += f"- **Recall**: {best_accuracy['recall']:.4f}\n"
    report += f"- **F1 Score**: {best_accuracy['f1_score']:.4f}\n"
    report += f"- **States**: {best_accuracy['states']}\n"
    report += f"- **Observations**: {best_accuracy['observations']}\n"
    report += f"- **Discretization Strategy**: {best_accuracy['discr_strategy']}\n"
    report += f"- **Direct States**: {best_accuracy['direct_states']}\n"
    report += f"- **Feature**: {best_accuracy['feature']}\n\n"

    report += "### Best by F1 Score\n\n"
    report += f"- **F1 Score**: {best_f1['f1_score']:.4f}\n"
    report += f"- **Accuracy**: {best_f1['accuracy']:.4f}\n"
    report += f"- **Precision**: {best_f1['precision']:.4f}\n"
    report += f"- **Recall**: {best_f1['recall']:.4f}\n"
    report += f"- **States**: {best_f1['states']}\n"
    report += f"- **Observations**: {best_f1['observations']}\n"
    report += f"- **Discretization Strategy**: {best_f1['discr_strategy']}\n"
    report += f"- **Direct States**: {best_f1['direct_states']}\n"
    report += f"- **Feature**: {best_f1['feature']}\n\n"

    # Add summary of all results
    report += "## All Results\n\n"

    # Create a table of all results sorted by accuracy
    report += "| States | Observations | Discretization | Direct States | Feature | Accuracy | Precision | Recall | F1 Score |\n"
    report += "|--------|--------------|----------------|---------------|---------|----------|-----------|--------|----------|\n"

    # Sort by accuracy descending
    sorted_df = df.sort_values('accuracy', ascending=False)

    for _, row in sorted_df.iterrows():
        report += f"| {row['states']} | {row['observations']} | {row['discr_strategy']} | {row['direct_states']} | {row['feature']} | "
        report += f"{row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} |\n"

    # Analysis of hyperparameter impact
    report += "\n## Hyperparameter Analysis\n\n"

    # Effect of number of states
    report += "### Effect of Number of States\n\n"
    states_analysis = df.groupby('states').agg({
        'accuracy': ['mean', 'std', 'max'],
        'precision': ['mean', 'std', 'max'],
        'recall': ['mean', 'std', 'max'],
        'f1_score': ['mean', 'std', 'max']
    }).reset_index()

    report += "| States | Avg Accuracy | Max Accuracy | Avg Precision | Max Precision | Avg Recall | Max Recall | Avg F1 | Max F1 |\n"
    report += "|--------|--------------|--------------|---------------|---------------|------------|------------|--------|--------|\n"

    for _, row in states_analysis.iterrows():
        report += f"| {row['states']} | {row[('accuracy', 'mean')]:.4f} | {row[('accuracy', 'max')]:.4f} | "
        report += f"{row[('precision', 'mean')]:.4f} | {row[('precision', 'max')]:.4f} | "
        report += f"{row[('recall', 'mean')]:.4f} | {row[('recall', 'max')]:.4f} | "
        report += f"{row[('f1_score', 'mean')]:.4f} | {row[('f1_score', 'max')]:.4f} |\n"

    # Effect of number of observations
    report += "\n### Effect of Number of Observations\n\n"
    obs_analysis = df.groupby('observations').agg({
        'accuracy': ['mean', 'std', 'max'],
        'precision': ['mean', 'std', 'max'],
        'recall': ['mean', 'std', 'max'],
        'f1_score': ['mean', 'std', 'max']
    }).reset_index()

    report += "| Observations | Avg Accuracy | Max Accuracy | Avg Precision | Max Precision | Avg Recall | Max Recall | Avg F1 | Max F1 |\n"
    report += "|--------------|--------------|--------------|---------------|---------------|------------|------------|--------|--------|\n"

    for _, row in obs_analysis.iterrows():
        report += f"| {row['observations']} | {row[('accuracy', 'mean')]:.4f} | {row[('accuracy', 'max')]:.4f} | "
        report += f"{row[('precision', 'mean')]:.4f} | {row[('precision', 'max')]:.4f} | "
        report += f"{row[('recall', 'mean')]:.4f} | {row[('recall', 'max')]:.4f} | "
        report += f"{row[('f1_score', 'mean')]:.4f} | {row[('f1_score', 'max')]:.4f} |\n"

    # Effect of discretization strategy
    report += "\n### Effect of Discretization Strategy\n\n"
    discr_analysis = df.groupby('discr_strategy').agg({
        'accuracy': ['mean', 'std', 'max'],
        'precision': ['mean', 'std', 'max'],
        'recall': ['mean', 'std', 'max'],
        'f1_score': ['mean', 'std', 'max']
    }).reset_index()

    report += "| Strategy | Avg Accuracy | Max Accuracy | Avg Precision | Max Precision | Avg Recall | Max Recall | Avg F1 | Max F1 |\n"
    report += "|----------|--------------|--------------|---------------|---------------|------------|------------|--------|--------|\n"

    for _, row in discr_analysis.iterrows():
        report += f"| {row['discr_strategy']} | {row[('accuracy', 'mean')]:.4f} | {row[('accuracy', 'max')]:.4f} | "
        report += f"{row[('precision', 'mean')]:.4f} | {row[('precision', 'max')]:.4f} | "
        report += f"{row[('recall', 'mean')]:.4f} | {row[('recall', 'max')]:.4f} | "
        report += f"{row[('f1_score', 'mean')]:.4f} | {row[('f1_score', 'max')]:.4f} |\n"

    # Effect of direct_states parameter
    report += "\n### Effect of Direct States Parameter\n\n"
    direct_analysis = df.groupby('direct_states').agg({
        'accuracy': ['mean', 'std', 'max'],
        'precision': ['mean', 'std', 'max'],
        'recall': ['mean', 'std', 'max'],
        'f1_score': ['mean', 'std', 'max']
    }).reset_index()

    report += "| Direct States | Avg Accuracy | Max Accuracy | Avg Precision | Max Precision | Avg Recall | Max Recall | Avg F1 | Max F1 |\n"
    report += "|--------------|--------------|--------------|---------------|---------------|------------|------------|--------|--------|\n"

    for _, row in direct_analysis.iterrows():
        report += f"| {row['direct_states']} | {row[('accuracy', 'mean')]:.4f} | {row[('accuracy', 'max')]:.4f} | "
        report += f"{row[('precision', 'mean')]:.4f} | {row[('precision', 'max')]:.4f} | "
        report += f"{row[('recall', 'mean')]:.4f} | {row[('recall', 'max')]:.4f} | "
        report += f"{row[('f1_score', 'mean')]:.4f} | {row[('f1_score', 'max')]:.4f} |\n"

    # Effect of feature
    report += "\n### Effect of Feature Selection\n\n"
    feature_analysis = df.groupby('feature').agg({
        'accuracy': ['mean', 'std', 'max'],
        'precision': ['mean', 'std', 'max'],
        'recall': ['mean', 'std', 'max'],
        'f1_score': ['mean', 'std', 'max']
    }).reset_index()

    report += "| Feature | Avg Accuracy | Max Accuracy | Avg Precision | Max Precision | Avg Recall | Max Recall | Avg F1 | Max F1 |\n"
    report += "|---------|--------------|--------------|---------------|---------------|------------|------------|--------|--------|\n"

    for _, row in feature_analysis.iterrows():
        report += f"| {row['feature']} | {row[('accuracy', 'mean')]:.4f} | {row[('accuracy', 'max')]:.4f} | "
        report += f"{row[('precision', 'mean')]:.4f} | {row[('precision', 'max')]:.4f} | "
        report += f"{row[('recall', 'mean')]:.4f} | {row[('recall', 'max')]:.4f} | "
        report += f"{row[('f1_score', 'mean')]:.4f} | {row[('f1_score', 'max')]:.4f} |\n"

    # Confusion Matrix for the best model
    report += "\n## Confusion Matrix for Best Model\n\n"

    # Get the confusion matrix from the best model by accuracy
    if 'confusion_matrix' in best_accuracy:
        cm = best_accuracy['confusion_matrix']
        report += "```\n"
        report += f"                 Predicted\n"
        report += f"                 Bear    Bull\n"
        report += f"Actual Bear      {cm[0,0]:<7} {cm[0,1]:<7}\n"
        report += f"       Bull      {cm[1,0]:<7} {cm[1,1]:<7}\n"
        report += "```\n\n"

    # Recommendations
    report += "## Recommendations\n\n"

    # Analyze trends to make recommendations
    best_states = sorted_df.iloc[0:3]['states'].mode()[0]
    best_obs = sorted_df.iloc[0:3]['observations'].mode()[0]
    best_strategy = sorted_df.iloc[0:3]['discr_strategy'].mode()[0]
    best_direct = sorted_df.iloc[0:3]['direct_states'].mode()[0]
    best_feature = sorted_df.iloc[0:3]['feature'].mode()[0]

    report += "Based on the hyperparameter optimization results, we recommend the following configuration:\n\n"
    report += f"- **States**: {best_states}\n"
    report += f"- **Observations**: {best_obs}\n"
    report += f"- **Discretization Strategy**: {best_strategy}\n"
    report += f"- **Direct States**: {best_direct}\n"
    report += f"- **Feature**: {best_feature}\n\n"

    # Trends and insights
    report += "### Key Insights\n\n"

    # Analyze trends in the data
    report += "1. **Number of States**: "
    if states_analysis[('accuracy', 'mean')].corr(states_analysis['states']) > 0:
        report += "Higher number of states tends to improve performance, suggesting the model benefits from capturing more market regimes.\n"
    else:
        report += "Having more states doesn't necessarily improve performance, suggesting simpler models can capture the essential market dynamics.\n"

    report += "2. **Number of Observations**: "
    if obs_analysis[('accuracy', 'mean')].corr(obs_analysis['observations']) > 0:
        report += "Increasing the number of observations generally improves model performance, allowing for finer-grained discretization of the input data.\n"
    else:
        report += "More observations doesn't necessarily lead to better performance, suggesting a balance between granularity and generalization.\n"

    report += "3. **Discretization Strategy**: "
    if best_strategy == 'kmeans':
        report += "K-means clustering tends to perform better than equal-width or equal-frequency binning, likely because it adapts to the natural distribution of the data.\n"
    elif best_strategy == 'equal_freq':
        report += "Equal-frequency binning performed well, ensuring balanced representation across bins.\n"
    else:
        report += "Equal-width binning worked best, suggesting linear discretization is appropriate for this data.\n"

    report += "4. **Direct States vs. Majority Voting**: "
    direct_better = direct_analysis.loc[direct_analysis['direct_states'] == True, ('accuracy', 'mean')].values[0] > \
        direct_analysis.loc[direct_analysis['direct_states']
                            == False, ('accuracy', 'mean')].values[0]
    if direct_better:
        report += "Direct state correlation performs better than majority voting, indicating clear relationships between hidden states and market regimes.\n"
    else:
        report += "Majority voting performed better than direct state correlation, suggesting that aggregating labels within states provides more robust classifications.\n"

    report += "\n## Conclusion\n\n"
    report += f"This hyperparameter optimization study has identified an optimal HMM configuration with {best_states} states, "
    report += f"{best_obs} observation bins, using {best_strategy} discretization on the {best_feature} feature. "
    report += f"This configuration achieved an accuracy of {best_accuracy['accuracy']:.4f} and an F1 score of {best_accuracy['f1_score']:.4f} "
    report += f"in classifying bull and bear market regimes.\n\n"
    report += f"The results demonstrate that Hidden Markov Models can effectively capture market regime dynamics, "
    report += f"providing a valuable tool for financial time series analysis and potentially for trading strategy development."

    return report


if __name__ == "__main__":
    print("Running HMM hyperparameter optimization...")

    # Run the grid search
    results = run_hyperparameter_grid_search()

    # Generate and save the report
    report = generate_report(results)
    with open('report.md', 'w') as f:
        f.write(report)

    print("Optimization complete. Report saved to 'report.md'")
