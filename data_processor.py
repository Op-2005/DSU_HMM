import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FinancialDataLoader(Dataset):
    def __init__(self, file_path, target_column, features, normalize=True, data=None, device=None):
        """
        Custom PyTorch Dataset for loading financial data.
        
        Parameters:
        -----------
        file_path: Path to the cleaned financial dataset.
        target_column: The column to predict (string or list of strings).
        features: List of feature columns to use.
        normalize: Whether to normalize the features.
        data: Optional pre-loaded DataFrame (used for train-test split).
        device: Device to place tensors on (cuda or cpu).
        """
        # Set the device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Process target column (handle both string and tuple/list formats)
        if isinstance(target_column, (list, tuple)):
            self.target_column = target_column[0]  # Extract first element
        else:
            self.target_column = target_column
        
        self.features = features
        self.normalize = normalize

        # Load data only if not provided
        if data is None:
            self.data = pd.read_csv(file_path)
            print(f"Loaded data from {file_path}, shape: {self.data.shape}")
        else:
            self.data = data.copy()

        # Clean column names (strip spaces)
        self.data.columns = self.data.columns.str.strip()

        # Drop NaN values
        subset_cols = [self.target_column] + features
        subset_cols = [col for col in subset_cols if col in self.data.columns]
        original_len = len(self.data)
        self.data.dropna(subset=subset_cols, inplace=True)
        dropped_rows = original_len - len(self.data)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with NaN values")

        # Convert to numpy arrays
        self.X = self.data[features].values.astype(np.float32)
        self.y = self.data[self.target_column].values.astype(np.float32)

        # Normalize features if required
        if self.normalize:
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0) + 1e-8  # Avoid division by zero
            self.X = (self.X - self.mean) / self.std
            print(f"Normalized features, mean: {self.mean}, std: {self.std}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load data as tensors and send them to the correct device (CUDA or CPU)
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32, device=self.device)
        return X_tensor, y_tensor

    def get_data_loader(self, batch_size=32, shuffle=True):
        """
        Returns a DataLoader for batch processing.
        
        Parameters:
        -----------
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        
        Returns:
        --------
        DataLoader instance
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def train_test_split(self, test_size=0.2, shuffle=True):
        """
        Splits the dataset into training and testing sets.
        
        Parameters:
        -----------
        test_size: Fraction of data to use for testing (0.0 - 1.0).
        shuffle: Whether to shuffle the dataset before splitting.
        
        Returns:
        --------
        Two FinancialDataLoader instances (train, test).
        """
        num_samples = len(self.data)
        print(f"Splitting data: {num_samples} samples with test_size={test_size}")

        # Shuffle indices if needed
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        # Split index
        split_idx = int(num_samples * (1 - test_size))
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]

        # Create training and testing datasets
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

def discretize_data(data, num_bins=10, strategy='equal_width'):
    """
    Discretize continuous data for HMM processing.
    
    Parameters:
    -----------
    data: Continuous data array
    num_bins: Number of discrete bins to create
    strategy: 'equal_width', 'equal_freq', or 'kmeans'
    
    Returns:
    --------
    Discretized data as indices
    """
    # Ensure data is 1D
    if len(data.shape) > 1 and data.shape[1] > 1:
        raise ValueError("discretize_data expects a 1D array")
    
    flat_data = data.flatten()
    
    if strategy == 'equal_width':
        # Equal width binning
        bins = np.linspace(np.min(flat_data), np.max(flat_data), num_bins + 1)
    elif strategy == 'equal_freq':
        # Equal frequency binning
        bins = np.percentile(flat_data, np.linspace(0, 100, num_bins + 1))
    elif strategy == 'kmeans':
        # K-means binning (requires scikit-learn)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(flat_data.reshape(-1, 1))
        centers = np.sort(kmeans.cluster_centers_.flatten())
        bins = np.concatenate([[-np.inf], (centers[:-1] + centers[1:]) / 2, [np.inf]])
    else:
        raise ValueError(f"Unknown discretization strategy: {strategy}")
    
    # Ensure unique bin edges
    bins = np.unique(bins)
    
    # Digitize the data
    discretized = np.digitize(flat_data, bins) - 1
    
    # Cap maximum state to num_bins-1
    discretized = np.minimum(discretized, num_bins - 1)
    
    # Reshape to original shape
    return discretized.reshape(data.shape)

def combine_features(data, features, method='first'):
    """
    Combine multiple features into a single feature for HMM.
    
    Parameters:
    -----------
    data: DataFrame or numpy array with features
    features: List of feature names or column indices
    method: 'first', 'mean', 'pca', or 'custom'
    
    Returns:
    --------
    Combined feature as 1D array
    """
    if method == 'first':
        # Just use the first feature
        if isinstance(data, pd.DataFrame):
            return data[features[0]].values
        else:
            return data[:, 0]
    elif method == 'mean':
        # Take the mean of features
        if isinstance(data, pd.DataFrame):
            return data[features].mean(axis=1).values
        else:
            return np.mean(data, axis=1)
    elif method == 'pca':
        # Use first principal component
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        if isinstance(data, pd.DataFrame):
            return pca.fit_transform(data[features].values).flatten()
        else:
            return pca.fit_transform(data).flatten()
    elif method == 'custom':
        # Define your custom feature combination logic here
        pass
    else:
        raise ValueError(f"Unknown feature combination method: {method}")