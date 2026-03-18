import torch
import numpy as np

class ModelTrainer:
    """
    Responsible  for managing the training loop, updating model weights,
    and recording epoch-level losses.

    Args:
        model (torch.nn.Module): The initialized PyTorch model.
        optimizer (torch.optim.Optimizer): The optimizer algorithm.
        criterion (torch.nn.modules.loss): The loss function.
        device (torch.device): CPU or CUDA device.
    """

    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.train_losses = []
        self.val_losses = []

    def train(self, epochs, train_loader, val_loader, evaluator):
        """
        Executes the training loop over the specified number of epochs.
        Utilizes the ModelEvaluator to check validation loss per epoch.
        """
        print(f"Starting training on device: {self.device}")
        for epoch in range(epochs):
            self.model.train()
            batch_losses = []

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())

            # Record average training loss
            train_loss = np.mean(batch_losses)
            self.train_losses.append(train_loss)

            # Use evaluator to get validation loss
            val_loss, _, _ = evaluator.evaluate(self.model, val_loader)
            self.val_losses.append(val_loss)

            print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')