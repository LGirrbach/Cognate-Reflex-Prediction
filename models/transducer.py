import torch
import torch.nn as nn

from torch import Tensor
from models.decoder import Decoder
from models.encoder import EncoderInput
from models.decoder import DecoderInput
from models.decoder import DecoderOutput
from models.encoder import CognateGridEncoder
from models.decoder import AutoregressiveLSTMDecoder
from models.decoder import NonAutoregressiveLSTMDecoder
from models.decoder import NonAutoregressiveFixedTauDecoder
from models.decoder import NonAutoregressivePositionalDecoder


def _decoder_factory(decoder_name: str, **kwargs) -> Decoder:
    if decoder_name == "autoregressive":
        return AutoregressiveLSTMDecoder(
            target_vocab_size=kwargs["target_vocab_size"], prediction_vocab_size=kwargs["prediction_vocab_size"],
            encoder_output_size=kwargs["encoder_output_size"], embedding_dim=kwargs["embedding_dim"],
            hidden_size=kwargs["hidden_size"], num_layers=kwargs["num_layers"], device=kwargs["device"],
            dropout=kwargs["dropout"], encoder_bridge=kwargs["encoder_bridge"]
        )

    elif decoder_name == "non-autoregressive-lstm":
        return NonAutoregressiveLSTMDecoder(
            prediction_vocab_size=kwargs["prediction_vocab_size"], encoder_output_size=kwargs["encoder_output_size"],
            hidden_size=kwargs["hidden_size"], num_layers=kwargs["num_layers"], device=kwargs["device"],
            dropout=kwargs["dropout"]
        )

    elif decoder_name == "non-autoregressive-position":
        return NonAutoregressivePositionalDecoder(
            prediction_vocab_size=kwargs["prediction_vocab_size"], encoder_output_size=kwargs["encoder_output_size"],
            hidden_size=kwargs["hidden_size"], device=kwargs["device"], dropout=kwargs["dropout"],
            embedding_dim=kwargs["embedding_dim"], max_predictions_per_symbol=kwargs["max_predictions_per_symbol"]
        )

    elif decoder_name == "non-autoregressive-fixed":
        return NonAutoregressiveFixedTauDecoder(
            prediction_vocab_size=kwargs["prediction_vocab_size"], encoder_output_size=kwargs["encoder_output_size"],
            hidden_size=kwargs["hidden_size"], device=kwargs["device"], dropout=kwargs["dropout"], tau=kwargs["tau"]
        )

    else:
        raise ValueError(f"Unknown Decoder: {decoder_name}")


class TransducerModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.encoder = CognateGridEncoder(
            source_vocab_size=kwargs["source_vocab_size"], language_vocab_size=kwargs["language_vocab_size"],
            embedding_dim=kwargs["embedding_dim"], device=kwargs["device"], hidden_size=kwargs["hidden_size"],
            num_layers=kwargs["num_layers"], dropout=kwargs["dropout"]
        )
        self.kwargs["encoder_output_size"] = self.encoder.output_size
        self.decoder = _decoder_factory(**self.kwargs)

    def get_params(self):
        return self.kwargs

    def encode(self, encoder_inputs: EncoderInput) -> Tensor:
        return self.encoder(encoder_inputs=encoder_inputs)

    def decode(self, decoder_inputs: DecoderInput) -> DecoderOutput:
        return self.decoder(decoder_inputs=decoder_inputs)

    def get_scores(self, encoder_inputs: EncoderInput, decoder_inputs: DecoderInput):
        # Encode source sequences
        encoder_outputs = self.encode(encoder_inputs=encoder_inputs)

        # Update decoder inputs
        decoder_inputs.encoder_outputs = encoder_outputs
        decoder_inputs.encoder_output_lengths = encoder_inputs.num_segments

        # Calculate scores
        decoder_outputs = self.decode(decoder_inputs=decoder_inputs)
        scores = torch.log_softmax(decoder_outputs.scores, dim=-1)

        return scores
