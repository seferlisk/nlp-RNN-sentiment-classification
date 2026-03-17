import torch.nn as nn
import torch.nn.functional as F

class StackedBiLSTMAttention(nn.Module):
    """
    A PyTorch module combining:
      - Trainable Embedding Layer
      - 2-layer (stacked) Bidirectional LSTM
      - Attention mechanism
      - Linear classification layer
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(StackedBiLSTMAttention, self).__init__()

        # 1. Trainable Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)

        # 2. Stacked Bi-LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

        # 3. Attention mechanism mapping Bi-LSTM outputs to a scalar score
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1)
        )

        # 4. Final fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length)

        # Pass integer sequences through embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Pass embeddings through LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # lstm_out: (batch, seq_len, hidden_dim * 2)

        # Apply attention to each time step
        energy = self.attn(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(energy, dim=1)  # (batch, seq_len, 1)

        # Compute weighted context vector
        context = (lstm_out * attention_weights).sum(dim=1)  # (batch, hidden_dim * 2)

        # Map context vector to class probabilities
        out = self.fc(context)  # (batch, output_dim)
        return out