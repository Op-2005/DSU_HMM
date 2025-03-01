import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from hmm_model import HiddenMarkovModel
from data_processor import FinancialDataLoader
# Set device to CUDA if available

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def discretize_data(data, num_states=10):
    bins = np.percentile(data, np.linspace(0, 100, num_states+1))
    bins = np.unique(bins)
    discretized = np.digitize(data, bins) - 1
    discretized = np.minimum(discretized, num_states-1)
    return discretized
def initialize_hmm_params(num_states, num_observations):
    T = np.ones((num_states, num_states)) / num_states
    T = T + np.random.uniform(0, 0.1, T.shape)
    T = T / T.sum(axis=1, keepdims=True)
    E = np.ones((num_states, num_observations)) / num_observations
    E = E + np.random.uniform(0, 0.1, E.shape)
    E = E / E.sum(axis=1, keepdims=True)
    T0 = np.ones(num_states) / num_states
    return T, E, T0
def train_hmm(X_train, num_states, num_observations, max_steps=20):
    T, E, T0 = initialize_hmm_params(num_states, num_observations)
    hmm = HiddenMarkovModel(T, E, T0, epsilon=0.001, maxStep=max_steps)
    obs_seq = X_train.astype(int)
    print("Training HMM...")
    T0, T, E, converged = hmm.Baum_Welch_EM(obs_seq)
    return hmm

def evaluate_hmm(hmm, X_test, y_test):
    states_seq, prob_scores = hmm.viterbi_inference(X_test)
    states_seq = states_seq.numpy()
    correlation = np.corrcoef(states_seq, y_test)[0, 1]
    return {"correlation": correlation, "states_seq": states_seq, "prob_scores": prob_scores.numpy()}

def main():
    file_path = "financial_data.csv"  # Replace with your file path
    target_column = "sp500 close",  # Replace with your target column
    features = ['sp500 open', 'sp500 high', 'sp500 low', 'sp500 volume']
    num_states = 3  # Number of hidden states in HMM
    num_observations = 10  # Number of discrete observation values
    batch_size = 64
    data_loader = FinancialDataLoader(file_path, target_column, features, normalize=True)
    train_dataset, test_dataset = data_loader.train_test_split(test_size=0.2)
    train_loader = train_dataset.get_data_loader(batch_size=batch_size)
    test_loader = test_dataset.get_data_loader(batch_size=batch_size, shuffle=False)
    X_train_list = []
    y_train_list = []
    for X_batch, y_batch in train_loader:
        X_train_list.append(X_batch.numpy())
        y_train_list.append(y_batch.numpy())
    X_train_full = np.vstack(X_train_list)
    y_train_full = np.concatenate(y_train_list)
    X_test_list = []
    y_test_list = []
    for X_batch, y_batch in test_loader:
        X_test_list.append(X_batch.numpy())
        y_test_list.append(y_batch.numpy())
    X_test_full = np.vstack(X_test_list)
    y_test_full = np.concatenate(y_test_list)
    feature_idx = 0
    X_train_discrete = discretize_data(X_train_full[:, feature_idx], num_observations)
    X_test_discrete = discretize_data(X_test_full[:, feature_idx], num_observations)
    hmm = train_hmm(X_train_discrete, num_states, num_observations)
    evaluation = evaluate_hmm(hmm, X_test_discrete, y_test_full)
    print(f"Correlation between inferred states and target: {evaluation['correlation']:.4f}")
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(evaluation['states_seq'], label='Inferred States')
    plt.title('Inferred Hidden States')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(y_test_full, label='Actual Target', color='orange')
    plt.title('Actual Target Values')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(X_test_full[:, feature_idx], label=f'Feature {features[feature_idx]}', color='green')
    plt.title(f'Original Feature: {features[feature_idx]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hmm_results.png')
    plt.show()
    print("Training and evaluation complete!")
if __name__ == "__main__":
    main()











