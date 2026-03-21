import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Handles evaluation logic, metrics generation, and result visualizations.
    Separating this from the trainer ensures cleaner, modular code.

    Args:
        criterion (torch.nn.modules.loss): The loss function used for evaluation.
        device (torch.device): CPU or CUDA device.
    """

    def __init__(self, criterion, device):
        self.criterion = criterion
        self.device = device

    def evaluate(self, model, data_loader):
        """
        Runs a forward pass over the data_loader without tracking gradients.

        Returns:
            tuple: (average_loss, true_labels_list, predicted_labels_list)
        """
        model.eval()
        batch_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                outputs = model(X_batch)
                loss = self.criterion(outputs, y_batch)

                batch_losses.append(loss.item())
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        return np.mean(batch_losses), all_targets, all_preds

    def plot_learning_curves(self, train_losses, val_losses):
        """Plots training vs validation loss over epochs."""
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Training Loss', marker='o')
        plt.plot(val_losses, label='Validation Loss', marker='o')
        plt.title('Learning Curves over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plots a seaborn heatmap of the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title('Validation Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()