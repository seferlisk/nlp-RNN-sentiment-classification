import torch
from torch.utils.data import Dataset


class TweetDataset(Dataset):
    """
    A custom PyTorch Dataset for the Airline Tweets.

    Args:
        X (numpy.ndarray): Padded integer sequences of shape (num_samples, max_length).
        y (numpy.ndarray): Integer labels of shape (num_samples,).
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]