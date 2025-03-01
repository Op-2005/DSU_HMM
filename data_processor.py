import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FinancialDataLoader(Dataset):
    def __init__(self, file_path, target_column, features, normalize=True, data=None):
        """
        Custom PyTorch Dataset for loading financial data.
        :param file_path: Path to the cleaned financial dataset.
        :param target_column: The column to predict.
        :param features: List of feature columns to use.
        :param normalize: Whether to normalize the features.
        :param data: Optional pre-loaded DataFrame (used for train-test split).
        """
        self.features = features
        self.target_column = target_column
        self.normalize = normalize

        # Load data only if not provided
        if data is None:
            self.data = pd.read_csv(file_path)
        else:
            self.data = data.copy()

        # Clean column names (strip spaces)
        self.data.columns = self.data.columns.str.strip()

        # Drop NaN values
        subset_cols = [target_column] + features
        subset_cols = [col for col in subset_cols if col in self.data.columns]
        self.data.dropna(subset=subset_cols, inplace=True)

        # Convert to numpy arrays
        self.X = self.data[features].values.astype(np.float32)
        print("printing before y")
        print(self.data.columns)

        # Ensure target_column is a string
        if isinstance(target_column, list):
            target_column = target_column[0]  # Extract first element if list

        self.y = self.data[target_column[0]].values.astype(np.float32)

        # Normalize features if required
        if self.normalize:
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0) + 1e-8  # Avoid division by zero
            self.X = (self.X - self.mean) / self.std

        # Detect available device (CUDA if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load data as tensors and send them to the correct device (CUDA or CPU)
        X_tensor = torch.tensor(self.X[idx]).to(self.device)
        y_tensor = torch.tensor(self.y[idx]).to(self.device)
        return X_tensor, y_tensor

    def get_data_loader(self, batch_size=32, shuffle=True):
        """
        Returns a DataLoader for batch processing.
        :param batch_size: Number of samples per batch.
        :param shuffle: Whether to shuffle the dataset.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def train_test_split(self, test_size=0.2, shuffle=True):
        """
        Splits the dataset into training and testing sets.
        :param test_size: Fraction of data to use for testing (0.0 - 1.0).
        :param shuffle: Whether to shuffle the dataset before splitting.
        :return: Two FinancialDataLoader instances (train, test).
        """
        num_samples = len(self.data)

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

        train_loader = FinancialDataLoader(
            file_path=None, target_column=self.target_column, features=self.features, 
            normalize=self.normalize, data=train_data
        )

        test_loader = FinancialDataLoader(
            file_path=None, target_column=self.target_column, features=self.features, 
            normalize=self.normalize, data=test_data
        )

        return train_loader, test_loader
