import torch
import torch.nn as nn


class Embedder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, dropout: float = 0.0):
        super(Embedder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0,
        )
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        return embedded
