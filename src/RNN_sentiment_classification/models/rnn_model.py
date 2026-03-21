import torch.nn as nn
import torch.nn.functional as F

class StackedBiLSTMAttention(nn.Module):
    """
    A Neural Network combining a Trainable Embedding, a Stacked Bidirectional LSTM,
    and an Attention Mechanism for sentiment classification.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the word embeddings.
        hidden_dim (int): Number of features in the LSTM hidden state.
        output_dim (int): Number of output classes (3 for sentiment).
        num_layers (int): Number of recurrent layers (default: 2).
        dropout (float): Dropout probability between LSTM layers (default: 0.2).
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(StackedBiLSTMAttention, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1)
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        # Attention
        energy = self.attn(lstm_out)
        attention_weights = F.softmax(energy, dim=1)
        context = (lstm_out * attention_weights).sum(dim=1)

        out = self.fc(context)
        return out