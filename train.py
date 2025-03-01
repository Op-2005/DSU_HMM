import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from hmm_model import HiddenMarkovModel
# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
class FinancialDataLoader(Dataset):
    def __init__(self, file_path, target_column, features, normalize=True):
        """
        Custom PyTorch Dataset for loading financial data.
        :param file_path: Path to the cleaned financial dataset.
        :param target_column: The column to predict.
        :param features: List of feature columns to use.
        :param normalize: Whether to normalize the features.
        """
        self.data = pd.read_csv(file_path)
        print(self.data.columns)
        print(self.data.head())
        self.features = features
        self.target_column = target_column
        self.normalize = normalize
        self.X = self.data[features].values.astype(np.float32)
        self.y = self.data[target_column].values.astype(np.float32)
        if self.normalize:
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0) + 1e-8
            self.X = (self.X - self.mean) / self.std
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    def get_data_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    def train_test_split(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        train_dataset = FinancialDatasetSplit(X_train, y_train)
        test_dataset = FinancialDatasetSplit(X_test, y_test)
        return train_dataset, test_dataset
class FinancialDatasetSplit(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    def get_data_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
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
    target_column = "price_movement"  # Replace with your target column
    features = ["volume", "open", "close", "high", "low"]  # Replace with your features
    num_states = 5  # Number of hidden states in HMM
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











