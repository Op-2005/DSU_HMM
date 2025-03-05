import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

# Import the HMM class
from hmm_model import HiddenMarkovModel

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_financial_data(file_path, normalize=True):
    """
    Load and preprocess financial data
    
    Parameters:
    -----------
    file_path: Path to CSV file
    normalize: Whether to normalize features
    
    Returns:
    --------
    data: Pandas DataFrame
    """
    print(f"Loading data from {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load data
    data = pd.read_csv(file_path)
    
    # Clean column names (strip spaces)
    data.columns = data.columns.str.strip()
    
    print(f"Loaded data with shape {data.shape}")
    print(f"Columns: {', '.join(data.columns)}")
    
    return data

def prepare_data(data, features, target_column, test_size=0.2, normalize=True):
    """
    Prepare data for HMM training
    
    Parameters:
    -----------
    data: Pandas DataFrame
    features: List of feature columns
    target_column: Target column for prediction
    test_size: Proportion of test data
    normalize: Whether to normalize features
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler_params
    """
    # Ensure target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data. Available columns: {', '.join(data.columns)}")
    
    # Ensure all feature columns exist
    for feat in features:
        if feat not in data.columns:
            raise ValueError(f"Feature column '{feat}' not found in data. Available columns: {', '.join(data.columns)}")
    
    # Drop rows with NaN in features or target
    subset_cols = [target_column] + features
    data_clean = data.dropna(subset=subset_cols)
    
    print(f"Dropped {len(data) - len(data_clean)} rows with NaN values")
    
    # Extract features and target
    X = data_clean[features].values.astype(np.float32)
    y = data_clean[target_column].values.astype(np.float32)
    
    # Split train and test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Normalize if requested
    scaler_params = {}
    if normalize:
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8  # Avoid division by zero
        
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
        scaler_params = {'mean': mean, 'std': std}
        print(f"Normalized features with mean={mean} and std={std}")
    
    return X_train, X_test, y_train, y_test, scaler_params

def discretize_data(data, num_bins=10, strategy='equal_freq'):
    """
    Discretize continuous data for HMM processing.
    
    Parameters:
    -----------
    data: Continuous data array
    num_bins: Number of discrete bins to create
    strategy: 'equal_width' or 'equal_freq'
    
    Returns:
    --------
    Discretized data as integers
    """
    flat_data = data.flatten() if len(data.shape) > 1 else data
    
    if strategy == 'equal_width':
        # Equal width binning
        min_val = np.min(flat_data)
        max_val = np.max(flat_data)
        bins = np.linspace(min_val, max_val, num_bins + 1)
    elif strategy == 'equal_freq':
        # Equal frequency binning
        bins = np.percentile(flat_data, np.linspace(0, 100, num_bins + 1))
    else:
        raise ValueError(f"Unknown discretization strategy: {strategy}")
    
    # Ensure unique bin edges
    bins = np.unique(bins)
    
    # Digitize the data (assign bin indices)
    discretized = np.digitize(flat_data, bins) - 1
    
    # Cap maximum state to num_bins-1
    discretized = np.minimum(discretized, num_bins - 1)
    
    # Reshape back to original shape if needed
    return discretized.reshape(data.shape).astype(np.int64)

def initialize_hmm_params(num_states, num_observations):
    """
    Initialize HMM parameters with some randomness to break symmetry.
    
    Parameters:
    -----------
    num_states: Number of hidden states
    num_observations: Number of possible observation values
    
    Returns:
    --------
    T, E, T0: Initial transition, emission, and initial state probabilities
    """
    # Initialize transition matrix (rows sum to 1)
    T = np.ones((num_states, num_states)) / num_states
    T = T + np.random.uniform(0, 0.1, T.shape)
    T = T / T.sum(axis=1, keepdims=True)
    
    # Initialize emission matrix (rows sum to 1)
    E = np.ones((num_observations, num_states)) / num_states
    E = E + np.random.uniform(0, 0.1, E.shape)
    E = E / E.sum(axis=1, keepdims=True)
    
    # Initialize initial state probabilities (sums to 1)
    T0 = np.ones(num_states) / num_states
    
    return T, E, T0

def train_hmm(X_train_discrete, num_states, num_observations, max_steps=20, precision='double'):
    """
    Train HMM model on financial data.
    
    Parameters:
    -----------
    X_train_discrete: Discretized training data (integers)
    num_states: Number of hidden states for the HMM
    num_observations: Number of possible observation values
    max_steps: Maximum number of Baum-Welch steps
    precision: 'double' or 'single' precision
    
    Returns:
    --------
    hmm: Trained HMM model
    """
    print("\n" + "="*50)
    print(f"Training HMM with {num_states} states and {num_observations} observations")
    print("="*50)
    
    # Initialize HMM parameters
    T, E, T0 = initialize_hmm_params(num_states, num_observations)
    
    # Create HMM model on specified device
    hmm = HiddenMarkovModel(T, E, T0, device=device, epsilon=0.001, maxStep=max_steps)
    
    # Convert to tensor if not already
    if not isinstance(X_train_discrete, torch.Tensor):
        X_train_discrete = torch.tensor(X_train_discrete, dtype=torch.int64, device=device)
    elif X_train_discrete.device != device:
        X_train_discrete = X_train_discrete.to(device)
    
    # Train model
    print(f"Running Baum-Welch EM algorithm on device: {device}")
    start_time = time.time()
    
    # Run a single epoch (EM step) if single_epoch is True
    T0, T, E, converged = hmm.Baum_Welch_EM(X_train_discrete)
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    if converged:
        print("HMM training converged!")
    else:
        print("HMM training did not converge within max_steps")
    
    return hmm

def evaluate_hmm(hmm, X_test_discrete, y_test):
    """
    Evaluate HMM model on test data.
    
    Parameters:
    -----------
    hmm: Trained HMM model
    X_test_discrete: Discretized test data
    y_test: Test target values
    
    Returns:
    --------
    metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*50)
    print("Evaluating HMM model")
    print("="*50)
    
    # Convert to tensor if not already
    if not isinstance(X_test_discrete, torch.Tensor):
        X_test_discrete = torch.tensor(X_test_discrete, dtype=torch.int64, device=device)
    elif X_test_discrete.device != device:
        X_test_discrete = X_test_discrete.to(device)
    
    # Get most likely state sequence for test data
    states_seq, state_probs = hmm.viterbi_inference(X_test_discrete)
    
    # Move tensors to CPU for further processing
    states_seq_np = states_seq.cpu().numpy()
    state_probs_np = state_probs.cpu().numpy()
    
    # Calculate correlation between inferred states and target values
    correlation = np.corrcoef(states_seq_np, y_test)[0, 1]
    print(f"Correlation between inferred states and target: {correlation:.4f}")
    
    return {
        "correlation": correlation,
        "states_seq": states_seq_np,
        "state_probs": state_probs_np
    }

def plot_results(evaluation, X_test, y_test, feature_idx=0, feature_name="Feature"):
    """
    Plot HMM results
    
    Parameters:
    -----------
    evaluation: Evaluation metrics from evaluate_hmm
    X_test: Test features
    y_test: Test target values
    feature_idx: Index of feature to plot
    feature_name: Name of feature to plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Inferred states
    plt.subplot(3, 1, 1)
    plt.plot(evaluation['states_seq'], label='Inferred States')
    plt.title('Inferred Hidden States')
    plt.legend()
    
    # Plot 2: Target values
    plt.subplot(3, 1, 2)
    plt.plot(y_test, label='Actual Target', color='orange')
    plt.title('Actual Target Values')
    plt.legend()
    
    # Plot 3: Original feature data
    plt.subplot(3, 1, 3)
    feature_data = X_test[:, feature_idx] if len(X_test.shape) > 1 else X_test
    plt.plot(feature_data, label=feature_name, color='green')
    plt.title(f'Original Feature: {feature_name}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hmm_results.png')
    print("Results plot saved to 'hmm_results.png'")
    plt.show()

def run_single_epoch_test(X_train_discrete, num_states, num_observations):
    """
    Run a single epoch of HMM training to test the implementation
    
    Parameters:
    -----------
    X_train_discrete: Discretized training data
    num_states: Number of hidden states for the HMM
    num_observations: Number of possible observation values
    
    Returns:
    --------
    hmm: HMM model after one epoch
    params: Updated parameters
    """
    print("\n" + "="*50)
    print("RUNNING SINGLE EPOCH TEST")
    print("="*50)
    
    # Initialize HMM parameters
    T, E, T0 = initialize_hmm_params(num_states, num_observations)
    
    # Create HMM model on specified device
    hmm = HiddenMarkovModel(T, E, T0, device=device, epsilon=0.001, maxStep=1)
    
    # Convert to tensor if not already
    if not isinstance(X_train_discrete, torch.Tensor):
        X_train_discrete = torch.tensor(X_train_discrete, dtype=torch.int64, device=device)
    elif X_train_discrete.device != device:
        X_train_discrete = X_train_discrete.to(device)
    
    # Initialize variables for forward-backward
    hmm.N = len(X_train_discrete)
    shape = [hmm.N, hmm.S]
    hmm.initialize_forw_back_variables(shape)
    
    # Run a single EM step
    print("Running single epoch (EM step)...")
    start_time = time.time()
    
    # Get emission probabilities for observation sequence
    obs_prob_seq = hmm.E[X_train_discrete]
    
    # Run forward-backward algorithm
    hmm.forward_backward(obs_prob_seq)
    
    # Re-estimate parameters
    new_T0, new_T = hmm.re_estimate_transition(X_train_discrete)
    new_E = hmm.re_estimate_emission(X_train_discrete)
    
    # Check convergence
    converged = hmm.check_convergence(new_T0, new_T, new_E)
    
    # Update parameters
    hmm.T0 = new_T0
    hmm.E = new_E
    hmm.T = new_T
    
    elapsed_time = time.time() - start_time
    print(f"Single epoch completed in {elapsed_time:.2f} seconds")
    
    # Print parameter updates
    print("\nInitial State Probabilities (T0):")
    print(hmm.T0.cpu().numpy())
    
    print("\nTransition Matrix (T) - First few rows:")
    print(hmm.T[:3].cpu().numpy())
    
    print("\nEmission Matrix (E) - First few rows:")
    print(hmm.E[:3].cpu().numpy())
    
    return hmm, {'T0': hmm.T0, 'T': hmm.T, 'E': hmm.E}

def main():
    """Main function to run the HMM training and evaluation"""
    # Configuration parameters
    file_path = "financial_data.csv"
    features = ["sp500 open", "sp500 high", "sp500 low", "sp500 close", "sp500 volume"]
    target_column = "sp500 close"
    
    num_states = 3             # Number of hidden states in HMM
    num_observations = 10      # Number of discrete states for observations
    test_size = 0.2            # Proportion of data for testing
    normalize = True           # Whether to normalize data
    
    # Run settings
    run_full_training = True   # Whether to run full training or just one epoch
    max_steps = 20             # Maximum number of Baum-Welch iterations
    
    try:
        # Load and prepare data
        data = load_financial_data(file_path, normalize=normalize)
        
        X_train, X_test, y_train, y_test, scaler_params = prepare_data(
            data, features, target_column, test_size=test_size, normalize=normalize
        )
        
        # Use the first feature for discretization
        feature_idx = 0
        feature_name = features[feature_idx]
        
        # Discretize data for HMM
        print(f"\nDiscretizing data using feature: {feature_name}")
        X_train_feature = X_train[:, feature_idx]
        X_test_feature = X_test[:, feature_idx]
        
        X_train_discrete = discretize_data(X_train_feature, num_bins=num_observations)
        X_test_discrete = discretize_data(X_test_feature, num_bins=num_observations)
        
        print(f"Discretized values range: {X_train_discrete.min()} to {X_train_discrete.max()}")
        
        if run_full_training:
            # Train HMM
            hmm = train_hmm(X_train_discrete, num_states, num_observations, max_steps=max_steps)
            
            # Evaluate HMM
            evaluation = evaluate_hmm(hmm, X_test_discrete, y_test)
        else:
            # Run single epoch test
            hmm, params = run_single_epoch_test(X_train_discrete, num_states, num_observations)
            
            # Evaluate HMM after single epoch
            evaluation = evaluate_hmm(hmm, X_test_discrete, y_test)
        
        # Plot results
        plot_results(evaluation, X_test, y_test, feature_idx=feature_idx, feature_name=feature_name)
        
        # Print GPU memory stats if using CUDA
        if torch.cuda.is_available():
            print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            torch.cuda.empty_cache()
        
        print("\nTraining and evaluation complete!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()