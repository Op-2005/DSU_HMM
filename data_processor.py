import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FinancialDataLoader(Dataset):
    def __init__(self, file_path, target_column, features, normalize=True, data=None, device=None):
        self.device = 'cpu'
        
        if isinstance(target_column, (list, tuple)):
            self.target_column = target_column[0]
        else:
            self.target_column = target_column
        
        self.features = features
        self.normalize = normalize

        if data is None:
            self.data = pd.read_csv(file_path)
            print(f"Loaded data from {file_path}, shape: {self.data.shape}")
        else:
            self.data = data.copy()

        self.data.columns = self.data.columns.str.strip()

        subset_cols = [self.target_column] + features
        subset_cols = [col for col in subset_cols if col in self.data.columns]
        original_len = len(self.data)
        self.data.dropna(subset=subset_cols, inplace=True)
        dropped_rows = original_len - len(self.data)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with NaN values")

        self.X = self.data[features].values.astype(np.float32)
        self.y = self.data[self.target_column].values.astype(np.float32)

        if self.normalize:
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0) + 1e-8
            self.X = (self.X - self.mean) / self.std
            print(f"Normalized features, mean: {self.mean}, std: {self.std}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        return X_tensor, y_tensor

    def get_data_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def train_test_split(self, test_size=0.2, shuffle=True):
        num_samples = len(self.data)
        print(f"Splitting data: {num_samples} samples with test_size={test_size}")

        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        split_idx = int(num_samples * (1 - test_size))
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]

        train_data = self.data.iloc[train_indices].copy()
        test_data = self.data.iloc[test_indices].copy()

        print(f"Train set: {len(train_data)} samples, Test set: {len(test_data)} samples")

        train_loader = FinancialDataLoader(
            file_path=None, target_column=self.target_column, features=self.features, 
            normalize=self.normalize, data=train_data, device=self.device
        )

        test_loader = FinancialDataLoader(
            file_path=None, target_column=self.target_column, features=self.features, 
            normalize=self.normalize, data=test_data, device=self.device
        )

        return train_loader, test_loader

    def add_log_returns(self, price_column):
        if price_column not in self.data.columns:
            raise ValueError(f"Column {price_column} not found in data")
        
        log_returns_col = f"{price_column}_log_return"
        self.data[log_returns_col] = np.log(self.data[price_column] / self.data[price_column].shift(1))
        self.data.dropna(subset=[log_returns_col], inplace=True)
        
        print(f"Added log returns column: {log_returns_col}")
        return log_returns_col

    def add_regime_labels(self, returns_column, threshold=0.0, window=None):
        if returns_column not in self.data.columns:
            raise ValueError(f"Column {returns_column} not found in data")
        
        label_col = "actual_label"
        
        if window is not None and window > 1:
            smoothed_returns = self.data[returns_column].rolling(window=window).mean()
            self.data[label_col] = (smoothed_returns > threshold).astype(int)
            print(f"Added regime labels using {window}-day smoothed returns")
        else:
            self.data[label_col] = (self.data[returns_column] > threshold).astype(int)
            print(f"Added regime labels using daily returns with threshold {threshold}")
        
        self.data.dropna(subset=[label_col], inplace=True)
        
        print(f"Added regime labels column: {label_col}")
        print(f"Bull market days: {self.data[label_col].sum()} ({self.data[label_col].mean()*100:.1f}%)")
        print(f"Bear market days: {(self.data[label_col] == 0).sum()} ({(1-self.data[label_col].mean())*100:.1f}%)")
        
        return label_col


def discretize_data(data, num_bins=10, strategy='equal_width'):
    if len(data.shape) > 1 and data.shape[1] > 1:
        raise ValueError("discretize_data expects a 1D array")
    
    flat_data = data.flatten()
    
    if strategy == 'equal_width':
        bins = np.linspace(np.min(flat_data), np.max(flat_data), num_bins + 1)
    elif strategy == 'equal_freq':
        bins = np.percentile(flat_data, np.linspace(0, 100, num_bins + 1))
    elif strategy == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(flat_data.reshape(-1, 1))
        centers = np.sort(kmeans.cluster_centers_.flatten())
        bins = np.concatenate([[-np.inf], (centers[:-1] + centers[1:]) / 2, [np.inf]])
    else:
        raise ValueError(f"Unknown discretization strategy: {strategy}")
    
    bins = np.unique(bins)
    discretized = np.digitize(flat_data, bins) - 1
    discretized = np.minimum(discretized, num_bins - 1)
    
    return discretized.reshape(data.shape)


def combine_features(data, features, method='first'):
    if method == 'first':
        if isinstance(data, pd.DataFrame):
            return data[features[0]].values
        else:
            return data[:, 0]
    elif method == 'mean':
        if isinstance(data, pd.DataFrame):
            return data[features].mean(axis=1).values
        else:
            return np.mean(data, axis=1)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        if isinstance(data, pd.DataFrame):
            return pca.fit_transform(data[features].values).flatten()
        else:
            return pca.fit_transform(data).flatten()
    elif method == 'custom':
        pass
    else:
        raise ValueError(f"Unknown feature combination method: {method}")


def map_bins_to_values(discretized_data, original_data, strategy='midpoint'):
    unique_bins = np.unique(discretized_data)
    bin_map = {}
    
    for bin_idx in unique_bins:
        bin_mask = (discretized_data == bin_idx)
        bin_data = original_data[bin_mask]
        
        if strategy == 'midpoint':
            bin_map[bin_idx] = (np.min(bin_data) + np.max(bin_data)) / 2
        elif strategy == 'mean':
            bin_map[bin_idx] = np.mean(bin_data)
        else:
            raise ValueError(f"Unknown mapping strategy: {strategy}")
    
    return np.array([bin_map[idx] for idx in discretized_data.flatten()]).reshape(discretized_data.shape)
