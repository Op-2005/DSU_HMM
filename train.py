import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_processor import FinancialDataLoader, discretize_data, map_bins_to_values 
from hmm_model import HiddenMarkovModel

np.random.seed(42)
torch.manual_seed(42)

device = 'cpu'
print(f"Using device: {device}")

def load_financial_data(file_path, normalize=True):
    print(f"Loading data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    
    print(f"Loaded data with shape {data.shape}")
    print(f"Columns: {', '.join(data.columns)}")
    
    return data

def prepare_data(data, features, target_column, test_size=0.2, normalize=True):
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data. Available columns: {', '.join(data.columns)}")
    
    for feat in features:
        if feat not in data.columns:
            raise ValueError(f"Feature column '{feat}' not found in data. Available columns: {', '.join(data.columns)}")
    
    subset_cols = [target_column] + features
    data_clean = data.dropna(subset=subset_cols)
    
    print(f"Dropped {len(data) - len(data_clean)} rows with NaN values")
    
    X = data_clean[features].values.astype(np.float32)
    y = data_clean[target_column].values.astype(np.float32)
    
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    scaler_params = {}
    if normalize:
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8
        
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
        scaler_params = {'mean': mean, 'std': std}
        print(f"Normalized features with mean={mean} and std={std}")
    
    return X_train, X_test, y_train, y_test, scaler_params

def initialize_hmm_params(num_states, num_observations):
    T = np.ones((num_states, num_states)) / num_states
    T = T + np.random.uniform(0, 0.1, T.shape)
    T = T / T.sum(axis=1, keepdims=True)
    
    E = np.ones((num_observations, num_states)) / num_states
    E = E + np.random.uniform(0, 0.1, E.shape)
    E = E / E.sum(axis=1, keepdims=True)
    
    T0 = np.ones(num_states) / num_states
    
    return T, E, T0

def train_hmm(X_train_discrete, num_states, num_observations, max_steps=20):
    print("\n" + "="*50)
    print(f"Training HMM with {num_states} states and {num_observations} observations")
    print("="*50)
    
    T, E, T0 = initialize_hmm_params(num_states, num_observations)
    
    hmm = HiddenMarkovModel(T, E, T0, device='cpu', epsilon=0.001, maxStep=max_steps)
    
    if not isinstance(X_train_discrete, torch.Tensor):
        X_train_discrete = torch.tensor(X_train_discrete, dtype=torch.int64)
    
    print(f"Running Baum-Welch EM algorithm")
    start_time = time.time()
    
    T0, T, E, converged = hmm.Baum_Welch_EM(X_train_discrete)
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    if converged:
        print("HMM training converged!")
    else:
        print("HMM training did not converge within max_steps")
    
    return hmm

def run_single_epoch_test(X_train_discrete, num_states, num_observations):
    print("\n" + "="*50)
    print("RUNNING SINGLE EPOCH TEST")
    print("="*50)
    
    T, E, T0 = initialize_hmm_params(num_states, num_observations)
    
    hmm = HiddenMarkovModel(T, E, T0, device='cpu', epsilon=0.001, maxStep=1)
    
    if not isinstance(X_train_discrete, torch.Tensor):
        X_train_discrete = torch.tensor(X_train_discrete, dtype=torch.int64)
    
    hmm.N = len(X_train_discrete)
    shape = [hmm.N, hmm.S]
    hmm.initialize_forw_back_variables(shape)
    
    print("Running single epoch (EM step)...")
    start_time = time.time()
    
    obs_prob_seq = hmm.E[X_train_discrete]
    
    hmm.forward_backward(obs_prob_seq)
    
    new_T0, new_T = hmm.re_estimate_transition(X_train_discrete)
    new_E = hmm.re_estimate_emission(X_train_discrete)
    
    converged = hmm.check_convergence(new_T0, new_T, new_E)
    
    hmm.T0 = new_T0
    hmm.E = new_E
    hmm.T = new_T
    
    elapsed_time = time.time() - start_time
    print(f"Single epoch completed in {elapsed_time:.2f} seconds")
    
    print("\nInitial State Probabilities (T0):")
    print(hmm.T0.numpy())
    
    print("\nTransition Matrix (T) - First few rows:")
    print(hmm.T[:3].numpy())
    
    print("\nEmission Matrix (E) - First few rows:")
    print(hmm.E[:3].numpy())
    
    return hmm, {'T0': hmm.T0, 'T': hmm.T, 'E': hmm.E}

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate HMM model for financial data')
    parser.add_argument('--mode', type=str, default='classification', 
                        choices=['classification', 'forecasting'],
                        help='Evaluation mode: classification or forecasting')
    parser.add_argument('--states', type=int, default=3, 
                        help='Number of hidden states in HMM')
    parser.add_argument('--observations', type=int, default=10,
                        help='Number of discrete observation bins')
    parser.add_argument('--feature', type=str, default='sp500 close',
                        help='Feature to use for training')
    parser.add_argument('--target', type=str, default='sp500 close',
                        help='Target to predict')
    parser.add_argument('--steps', type=int, default=20,
                        help='Maximum number of Baum-Welch steps')
    parser.add_argument('--single-epoch', action='store_true',
                        help='Run only a single epoch test')
    return parser.parse_args()

def main():
    args = parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "financial_data.csv")
    test_size = 0.2
    normalize = True
    
    print(f"Running in {args.mode} mode with {args.states} states and {args.observations} observation bins")
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = load_financial_data(file_path)
        
        data_loader = FinancialDataLoader(
            file_path=None, 
            target_column=args.target,
            features=[args.feature], 
            normalize=normalize,
            data=data
        )
        
        log_returns_col = data_loader.add_log_returns(args.target)
        
        label_col = data_loader.add_regime_labels(log_returns_col, threshold=0.0, window=5)
        
        train_loader, test_loader = data_loader.train_test_split(test_size=test_size)
        
        X_train = train_loader.data[args.feature].values
        X_test = test_loader.data[args.feature].values
        y_train = train_loader.data[log_returns_col].values
        y_test = test_loader.data[log_returns_col].values
        train_labels = train_loader.data[label_col].values
        test_labels = test_loader.data[label_col].values
        
        X_train_discrete = discretize_data(y_train, num_bins=args.observations, strategy='equal_freq')
        X_test_discrete = discretize_data(y_test, num_bins=args.observations, strategy='equal_freq')
        
        if args.mode == 'forecasting':
            unique_bins = np.unique(X_train_discrete)
            bin_values = {}
            
            for bin_idx in unique_bins:
                bin_mask = (X_train_discrete == bin_idx)
                bin_values[bin_idx] = np.mean(y_train[bin_mask])
            
            obs_map = np.array([bin_values.get(i, 0) for i in range(args.observations)])
            print(f"Created observation map: {obs_map}")
        else:
            obs_map = None
        
        if args.single_epoch:
            hmm, params = run_single_epoch_test(X_train_discrete, args.states, args.observations)
        else:
            T, E, T0 = initialize_hmm_params(args.states, args.observations)
            hmm = HiddenMarkovModel(T, E, T0, device='cpu', maxStep=args.steps)
            
            T0, T, E, converged = hmm.Baum_Welch_EM(X_train_discrete)
        
        print(f"\nEvaluating HMM in {args.mode} mode...")
        eval_metrics = hmm.evaluate(
            X_test_discrete, 
            mode=args.mode,
            actual_values=y_test,
            actual_labels=test_labels,
            observation_map=obs_map
        )
        
        print("\n" + "="*50)
        print("HMM MODEL EVALUATION REPORT")
        print("="*50)
        
        if args.mode == 'classification':
            print(f"Classification Metrics:")
            print(f"  Accuracy:  {eval_metrics['accuracy']:.4f}")
            print(f"  Precision: {eval_metrics['precision']:.4f}")
            print(f"  Recall:    {eval_metrics['recall']:.4f}")
            print(f"  F1 Score:  {eval_metrics['f1_score']:.4f}")
            
            print("\nConfusion Matrix:")
            print(eval_metrics['confusion_matrix'])
            
            print("\nState Interpretations:")
            for state, interp in eval_metrics['state_interpretations'].items():
                print(f"  State {state}: {interp['type']}")
                print(f"    Bull Ratio: {interp['bull_ratio']:.2f}")
                print(f"    Mean Return: {interp['mean']:.6f}")
                print(f"    Std Deviation: {interp['std']:.6f}")
        
        else:
            print(f"Forecasting Metrics:")
            print(f"  MSE:         {eval_metrics['mse']:.6f}")
            print(f"  MAE:         {eval_metrics['mae']:.6f}")
            print(f"  Correlation: {eval_metrics['correlation']:.4f}")
        
        plt.figure(figsize=(15, 12))
        
        if args.mode == 'classification':
            plt.subplot(3, 1, 1)
            plt.plot(X_test, label=args.feature, color='blue')
            plt.title(f'{args.feature} with Inferred Market Regimes')
            
            states = eval_metrics['states_seq']
            unique_states = np.unique(states)
            colors = ['red', 'green', 'yellow', 'orange', 'purple']
            
            for i, state in enumerate(unique_states):
                state_interp = eval_metrics['state_interpretations'][state]
                state_type = state_interp['type']
                mask = (states == state)
                plt.fill_between(range(len(X_test)), np.min(X_test), np.max(X_test), 
                                 where=mask, alpha=0.3, color=colors[i % len(colors)],
                                 label=f"State {state}: {state_type}")
            plt.legend(loc='upper left')
            
            plt.subplot(3, 1, 2)
            plt.plot(y_test, label='Log Returns', color='blue')
            plt.scatter(range(len(y_test)), y_test, c=eval_metrics['predicted_labels'], 
                        cmap='coolwarm', alpha=0.6, label='Predicted Labels')
            plt.title('Log Returns with Bull/Bear Classifications')
            plt.axhline(y=0, color='black', linestyle='--')
            plt.legend()
            
            plt.subplot(3, 1, 3)
            cm = eval_metrics['confusion_matrix']
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bear', 'Bull'])
            disp.plot(ax=plt.gca(), cmap='Blues', values_format='.0f')
            plt.title('Confusion Matrix (Bear vs Bull Classification)')
            
        else:
            plt.subplot(3, 1, 1)
            plt.plot(eval_metrics['actual_next_values'], label='Actual Returns', color='blue')
            plt.plot(eval_metrics['forecasts'], label='Predicted Returns', color='red')
            plt.title('One-Step-Ahead Return Forecasts')
            plt.axhline(y=0, color='black', linestyle='--')
            plt.legend()
            
            plt.subplot(3, 1, 2)
            plt.scatter(eval_metrics['actual_next_values'], eval_metrics['forecasts'], alpha=0.5)
            plt.axhline(y=0, color='black', linestyle='--')
            plt.axvline(x=0, color='black', linestyle='--')
            plt.title(f'Actual vs Predicted Returns (Correlation: {eval_metrics["correlation"]:.4f})')
            plt.xlabel('Actual Returns')
            plt.ylabel('Predicted Returns')
            
            plt.subplot(3, 1, 3)
            plt.plot(X_test, label=args.feature, color='blue')
            plt.title(f'{args.feature} with Inferred Market Regimes')
            
            states = eval_metrics['states_seq']
            unique_states = np.unique(states)
            colors = ['red', 'green', 'yellow', 'orange', 'purple']
            
            for i, state in enumerate(unique_states):
                mask = (states == state)
                plt.fill_between(range(len(X_test)), np.min(X_test), np.max(X_test), 
                                 where=mask, alpha=0.3, color=colors[i % len(colors)],
                                 label=f"State {state}")
            plt.legend(loc='upper left')
        
        plt.tight_layout()
        output_file = f'hmm_{args.mode}_results.png'
        plt.savefig(output_file)
        print(f"\nResults plot saved to '{output_file}'")
        
        model_file = f'hmm_{args.mode}_model.pt'
        hmm.save_model(model_file)
        print(f"Model saved to '{model_file}'")

        # Now load the model back from disk to inspect parameters
        print("\nLoading the model back from disk to inspect parameters...")
        loaded_hmm = HiddenMarkovModel.load_model(model_file)

        print("\n=== HMM Model Parameters ===")
        print("Initial State Probabilities (T0):")
        print(loaded_hmm.T0)

        print("\nTransition Matrix (T):")
        print(loaded_hmm.T)

        print("\nEmission Matrix (E):")
        print(loaded_hmm.E)
        print("============================")

        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
