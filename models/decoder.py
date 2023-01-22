import torch
import torch.nn as nn

from abc import ABC
from typing import Tuple
from torch import Tensor
from containers import DecoderInput
from containers import DecoderOutput
from models.embedder import Embedder
from util import _max_over_time_pooling
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


def non_autoregressive_make_decoder_output(scores: Tensor) -> DecoderOutput:
    return DecoderOutput(scores=scores, old_hidden_states=None, new_hidden_states=None)


class Decoder(nn.Module, ABC):
    def __init__(self, prediction_vocab_size: int, encoder_output_size: int, hidden_size: int, dropout: float,
                 device: torch.device):
        super().__init__()

        self.prediction_vocab_size = prediction_vocab_size
        self.encoder_output_size = encoder_output_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.prediction_vocab_size)
        )

    def forward(self, decoder_inputs: DecoderInput) -> DecoderOutput:
        raise NotImplementedError


class AutoregressiveLSTMDecoder(Decoder):
    def __init__(self, target_vocab_size: int, prediction_vocab_size: int, encoder_output_size: int, embedding_dim: int,
                 hidden_size: int, num_layers: int, device: torch.device, dropout: float = 0.0,
                 encoder_bridge: bool = False):
        super(AutoregressiveLSTMDecoder, self).__init__(
            prediction_vocab_size=prediction_vocab_size, encoder_output_size=encoder_output_size,
            hidden_size=hidden_size, dropout=dropout, device=device
        )

        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder_bridge = encoder_bridge

        # Embedder
        self.decoder_embedder = Embedder(
            vocab_size=self.target_vocab_size, embedding_dim=self.embedding_dim, dropout=self.dropout
        )

        # Autoregressive LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True,
            dropout=(0.0 if self.num_layers == 1 else self.dropout)
        )

        # Encoder Bridge + Optional trainable first/last hidden states
        if self.encoder_bridge:
            self._encoder_bridge = nn.Linear(self.encoder_output_size, 2 * self.num_layers * self.hidden_size)
        else:
            # Initialise trainable hidden state initialisations
            self.h_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
            self.c_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))

        # Projection of concatenated encoder/decoder states
        self.classifier_input_projection = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.encoder_output_size + self.hidden_size, self.hidden_size),
            nn.GELU()
        )

    def _get_hidden(self, encoder_outputs: Tensor, encoder_lengths: Tensor, hidden: Tuple[Tensor, Tensor]):
        if hidden is not None:
            return hidden

        elif self.encoder_bridge:
            pooled_encoder_outputs = _max_over_time_pooling(encoder_outputs, encoder_lengths)
            pooled_encoder_outputs = self._encoder_bridge(pooled_encoder_outputs)
            batch_size = pooled_encoder_outputs.shape[0]
            pooled_encoder_outputs = pooled_encoder_outputs.reshape(batch_size, 2, self.num_layers, self.hidden_size)
            h_0 = pooled_encoder_outputs[:, 0].contiguous().transpose(0, 1)
            c_0 = pooled_encoder_outputs[:, 1].contiguous().transpose(0, 1)

            return h_0, c_0

        else:
            # Prepare hidden states
            batch_size = encoder_outputs.shape[0]
            h_0 = self.h_0.tile((1, batch_size, 1))
            c_0 = self.c_0.tile((1, batch_size, 1))

            return h_0, c_0

    def run_autoregressive_lstm(self, decoder_inputs: Tensor, decoder_input_lengths: Tensor, encoder_outputs: Tensor,
                                encoder_output_lengths: Tensor, hidden: Tuple[Tensor, Tensor] = None):
        # Pack sequence
        lengths = torch.clamp(decoder_input_lengths, 1)  # Enforce all lengths are >= 1 (required by pytorch)
        inputs = pack_padded_sequence(decoder_inputs, lengths, batch_first=True, enforce_sorted=False)

        # Initialise hidden states
        h_0, c_0 = self._get_hidden(encoder_outputs, encoder_output_lengths, hidden)
        old_hidden = (h_0, c_0)

        # Apply LSTM
        encoded, new_hidden = self.lstm(inputs, old_hidden)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True)

        return encoded, (old_hidden, new_hidden)

    def forward(self, decoder_inputs: DecoderInput) -> DecoderOutput:
        # Embed previous predictions
        previous_predictions = decoder_inputs.decoder_inputs.to(self.device)
        decoder_embedded = self.decoder_embedder(previous_predictions)

        # Run Autoregressive LSTM
        target_encodings, (old_hidden, new_hidden) = self.run_autoregressive_lstm(
            decoder_embedded, decoder_inputs.decoder_input_lengths, decoder_inputs.encoder_outputs,
            decoder_inputs.encoder_output_lengths, hidden=decoder_inputs.hidden_states
        )

        # Make all combinations of target symbols and source symbols
        source_encodings = decoder_inputs.encoder_outputs
        batch, h_features = target_encodings.shape[0], target_encodings.shape[2]
        timesteps_decoder = target_encodings.shape[1]
        timesteps_encoder = source_encodings.shape[1]

        decoder_outputs = target_encodings.unsqueeze(2)
        decoder_outputs = decoder_outputs.expand(batch, timesteps_decoder, timesteps_encoder, self.hidden_size)
        encoder_outputs = source_encodings.unsqueeze(2)
        encoder_outputs = encoder_outputs.expand(batch, timesteps_encoder, timesteps_decoder, self.encoder_output_size)
        encoder_outputs = encoder_outputs.transpose(1, 2)

        classifier_inputs = torch.cat([encoder_outputs, decoder_outputs], dim=-1)
        classifier_inputs = self.classifier_input_projection(classifier_inputs)
        scores = self.classifier(classifier_inputs)
        scores = scores.transpose(1, 2)

        return DecoderOutput(scores=scores, old_hidden_states=old_hidden, new_hidden_states=new_hidden)


class NonAutoregressiveLSTMDecoder(Decoder):
    def __init__(self, prediction_vocab_size: int, encoder_output_size: int, hidden_size: int, num_layers: int,
                 dropout: float, device: torch.device):
        super(NonAutoregressiveLSTMDecoder, self).__init__(
            prediction_vocab_size=prediction_vocab_size, encoder_output_size=encoder_output_size,
            hidden_size=hidden_size, dropout=dropout, device=device
        )

        self.num_layers = num_layers

        # Initialise LSTM
        self.decoder = nn.LSTM(
            input_size=self.encoder_output_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
            bias=True, bidirectional=False, batch_first=True, dropout=(0.0 if self.num_layers == 1 else self.dropout),
            proj_size=0
        )

        # Initialise Trainable first/last Hidden States
        self.h_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
        self.c_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, decoder_inputs: DecoderInput) -> DecoderOutput:
        # Unpack arguments
        encoder_outputs = decoder_inputs.encoder_outputs
        tau = decoder_inputs.tau

        # Make input sequence
        batch, timesteps, num_features = decoder_inputs.encoder_outputs.shape

        decoder_inputs = encoder_outputs.reshape(-1, num_features).unsqueeze(1)
        decoder_inputs = decoder_inputs.expand((decoder_inputs.shape[0], tau, num_features))

        hidden = (
            self.h_0.expand((self.num_layers, batch * timesteps, self.hidden_size)).contiguous(),
            self.c_0.expand((self.num_layers, batch * timesteps, self.hidden_size)).contiguous()
        )

        decoder_outputs, _ = self.decoder(decoder_inputs.contiguous(), hidden)
        decoder_outputs = decoder_outputs.reshape(batch, timesteps, tau, self.hidden_size)
        scores = self.classifier(decoder_outputs)

        return non_autoregressive_make_decoder_output(scores=scores)


class NonAutoregressivePositionalDecoder(Decoder):
    def __init__(self, prediction_vocab_size: int, encoder_output_size: int, hidden_size: int, dropout: float,
                 device: torch.device, embedding_dim: int, max_predictions_per_symbol: int):
        super(NonAutoregressivePositionalDecoder, self).__init__(
            prediction_vocab_size=prediction_vocab_size, encoder_output_size=encoder_output_size,
            hidden_size=hidden_size, dropout=dropout, device=device
        )

        self.embedding_dim = embedding_dim
        self.max_predictions_per_symbol = max_predictions_per_symbol

        self.positions = nn.Parameter(torch.zeros(1, self.max_predictions_per_symbol, self.embedding_dim))
        self.classifier_input_projection = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.encoder_output_size + self.embedding_dim, self.hidden_size),
            nn.GELU()
        )

    def forward(self, decoder_inputs: DecoderInput) -> DecoderOutput:
        # Unpack arguments
        encoder_outputs = decoder_inputs.encoder_outputs
        tau = decoder_inputs.tau

        if tau > self.max_predictions_per_symbol:
            tau = self.max_predictions_per_symbol

        # Make input sequence
        batch, timesteps, num_features = encoder_outputs.shape

        positions = self.positions[:, :tau, :].expand((batch * timesteps, tau, self.embedding_dim))
        positions = positions.contiguous()

        decoder_inputs = encoder_outputs.reshape(-1, num_features).unsqueeze(1)
        decoder_inputs = decoder_inputs.expand((decoder_inputs.shape[0], tau, num_features))
        decoder_inputs = decoder_inputs.contiguous()

        classifier_inputs = torch.cat([decoder_inputs, positions], dim=-1)
        classifier_inputs = self.classifier_input_projection(classifier_inputs)
        scores = self.classifier(classifier_inputs)
        scores = scores.reshape(batch, timesteps, tau, self.prediction_vocab_size)

        return non_autoregressive_make_decoder_output(scores=scores)


class NonAutoregressiveFixedTauDecoder(Decoder):
    def __init__(self, prediction_vocab_size: int, encoder_output_size: int, hidden_size: int, dropout: float,
                 device: torch.device, tau: int):
        super(NonAutoregressiveFixedTauDecoder, self).__init__(
            prediction_vocab_size=tau * prediction_vocab_size, encoder_output_size=encoder_output_size,
            hidden_size=hidden_size, dropout=dropout, device=device
        )
        self.tau = tau
        self.target_vocab_size = prediction_vocab_size

        self.classifier_input_projection = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.encoder_output_size, self.hidden_size),
            nn.GELU()
        )

    def forward(self, decoder_inputs: DecoderInput) -> DecoderOutput:
        scores = self.classifier_input_projection(decoder_inputs.encoder_outputs)
        scores = self.classifier(scores)
        scores = scores.reshape(scores.shape[0], scores.shape[1], self.tau, self.target_vocab_size)
        return non_autoregressive_make_decoder_output(scores=scores)
