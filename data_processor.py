import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
        self.features = features
        self.target_column = target_column
        self.normalize = normalize
        # Drop NaN values
        self.data.dropna(subset=[target_column] + features, inplace=True)
        # Convert to numpy arrays
        self.X = self.data[features].values.astype(np.float32)
        self.y = self.data[target_column].values.astype(np.float32)
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