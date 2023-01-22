import torch

from typing import List
from containers import Batch
from models import TransducerModel
from containers import AlignmentPosition
from containers import TransducerPrediction
from vocabulary import TransducerVocabulary
from actions import Deletion, Insertion, Substitution


def non_autoregressive_inference(model: TransducerModel, batch: Batch, target_vocabulary: TransducerVocabulary,
                                 max_predictions_per_symbol: int = 20) -> List[TransducerPrediction]:
    model = model.eval()
    model = model.to(model.kwargs["device"])

    # Prepare Model Inputs
    tau: int = model.kwargs.get("tau", None)
    if tau is None or tau > max_predictions_per_symbol:
        tau = max_predictions_per_symbol

    encoder_inputs = batch.encoder_inputs
    decoder_inputs = batch.decoder_inputs
    decoder_inputs.tau = tau

    # Get predictions
    with torch.no_grad():
        scores = model.get_scores(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        # scores shape: [batch x #source symbols x tau x #prediction symbols]
        batch_size, n_source_symbols, realised_tau, n_predictable_symbols = scores.shape
        assert tau == realised_tau
        scores = scores.reshape(batch_size, n_source_symbols * realised_tau, n_predictable_symbols)
        predictions = scores.argmax(dim=-1).detach().cpu()

    # Decode predictions
    hypotheses = []

    for cognate_set, predicted_action_idx in zip(batch.raw_sources, predictions):
        # Select only relevant predictions
        predicted_action_idx = predicted_action_idx[:tau * len(cognate_set)]
        predicted_action_idx = predicted_action_idx.reshape(-1, tau)
        predicted_action_idx = predicted_action_idx.tolist()

        action_history = []
        sequence_prediction = []

        for symbol, predicted_actions in zip(cognate_set, predicted_action_idx):
            current_actions = []
            current_predictions = []

            for action_idx in predicted_actions:
                action = target_vocabulary[action_idx]
                current_actions.append(action)

                if isinstance(action, Insertion):
                    current_predictions.append(action.token)

                elif isinstance(action, Substitution):
                    current_predictions.append(action.token)
                    break

                elif isinstance(action, Deletion):
                    break

            action_history.append({"symbol": symbol, "actions": current_actions, "predictions": current_predictions})
            sequence_prediction.extend(current_predictions)

        hypotheses.append((sequence_prediction, action_history))

    # Reformat hypotheses
    predictions = []
    for prediction, alignment in hypotheses:
        alignment = [
            AlignmentPosition(
                symbol=position["symbol"], actions=position["actions"], predictions=position["predictions"]
            ) for position in alignment
        ]
        predictions.append(TransducerPrediction(prediction=prediction, alignment=alignment))

    return predictions
