import torch
import torch.nn as nn

from torch import Tensor
from containers import EncoderInput
from models.embedder import Embedder
from models.bilstm import BiLSTMEncoder
from util import _max_over_time_pooling


class CognateGridEncoder(nn.Module):
    def __init__(self,  source_vocab_size: int, language_vocab_size: int, embedding_dim: int = 128,
                 device: torch.device = torch.device("cpu"), hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.0):
        super(CognateGridEncoder, self).__init__()

        self.source_vocab_size = source_vocab_size
        self.language_vocab_size = language_vocab_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.output_size = self.hidden_size + self.embedding_dim

        # Embedders
        self.cognate_embedder = Embedder(
            vocab_size=self.source_vocab_size, embedding_dim=self.embedding_dim, dropout=self.dropout
        )
        self.language_embedder = Embedder(
            vocab_size=self.language_vocab_size, embedding_dim=self.embedding_dim, dropout=self.dropout
        )

        # Encoders
        self.cognate_encoder = BiLSTMEncoder(
            input_size=2 * self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout, reduce_dim=False
        )
        self.segment_encoder = BiLSTMEncoder(
            input_size=2 * self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout, reduce_dim=True
        )

        self.gelu = nn.GELU()

    def forward(self, encoder_inputs: EncoderInput) -> Tensor:
        # Move all tensors to correct device
        cognate_sets = encoder_inputs.cognate_sets.to(self.device)
        source_languages = encoder_inputs.source_languages.to(self.device)
        target_languages = encoder_inputs.target_languages.to(self.device)

        # cognate_sets shape: [batch x #languages x #segments]
        batch, n_languages, n_segments = cognate_sets.shape

        # Embed Cognates + Source Languages
        embedded_cognate_sets = self.cognate_embedder(cognate_sets)
        embedded_source_languages = self.language_embedder(source_languages)

        # Concatenate Cognate + Language Embeddings
        embedded_source_languages = embedded_source_languages.reshape(batch, n_languages, 1, self.embedding_dim)
        embedded_source_languages = embedded_source_languages.repeat(1, 1, n_segments, 1)
        embedded_cognate_sets = torch.cat([embedded_cognate_sets, embedded_source_languages], dim=3)

        # Encode Cognates
        cognate_encoder_input = embedded_cognate_sets.reshape(-1, n_segments, 2 * self.embedding_dim)
        num_segments = encoder_inputs.num_segments.unsqueeze(1).repeat(1, n_languages).flatten()
        encoded_cognates = self.cognate_encoder(cognate_encoder_input, num_segments)
        encoded_cognates = encoded_cognates.reshape(batch, n_languages, n_segments, 2 * self.hidden_size)
        encoded_cognates = self.gelu(encoded_cognates)

        # Encode Segments
        segment_encoder_input = encoded_cognates.transpose(1, 2)
        segment_encoder_input = segment_encoder_input.reshape(-1, n_languages, 2 * self.hidden_size)
        num_languages = encoder_inputs.num_languages.unsqueeze(1).repeat(1, n_segments).flatten()

        encoded_segments = self.segment_encoder(segment_encoder_input, num_languages)
        encoded_segments = _max_over_time_pooling(encoded_segments, num_languages)
        encoded_segments = encoded_segments.reshape(batch, n_segments, self.hidden_size)
        encoded_segments = self.gelu(encoded_segments)

        # Embed Target Languages
        embedded_target_languages = self.language_embedder(target_languages)

        # Concatenate Encoded Segments + Target Languages
        embedded_target_languages = embedded_target_languages.unsqueeze(1)
        embedded_target_languages = embedded_target_languages.repeat(1, n_segments, 1)
        encoded_segments = torch.cat([encoded_segments, embedded_target_languages], dim=-1)

        return encoded_segments
