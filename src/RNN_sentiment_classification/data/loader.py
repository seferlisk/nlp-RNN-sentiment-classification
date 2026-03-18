import torch
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    """
    The PyTorch wrapper that allows the DataLoader to iterate
    through the preprocessed features and labels in batches.

    Args:
        X (numpy.ndarray): Padded integer sequences of shape (num_samples, max_length).
        y (numpy.ndarray): Integer labels of shape (num_samples,).
    """
    def __init__(self, X, y):
        # as_tensor is efficient: it avoids copying if the input is already a tensor
        # .long() ensures the data type is correct for Embedding and CrossEntropyLoss
        self.X = torch.as_tensor(X).long()
        self.y = torch.as_tensor(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]