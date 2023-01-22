import torch
import numpy as np

from typing import List
from copy import deepcopy
from containers import Beam
from containers import Batch
from models import TransducerModel
from containers import AlignmentPosition
from containers import TransducerPrediction
from vocabulary import TransducerVocabulary
from actions import Deletion, Insertion, Substitution, Noop


def autoregressive_greedy_sampling(model: TransducerModel, batch: Batch, target_vocabulary: TransducerVocabulary,
                                   max_decoding_length: int = 70) -> List[TransducerPrediction]:
    model = model.eval()
    model = model.to(model.kwargs["device"])

    sequences = batch.raw_sources
    source_lengths = batch.source_num_segments.cpu().tolist()

    # Run encoder
    with torch.no_grad():
        encoder_outputs = model.encode(batch.encoder_inputs)

    decoder_inputs = batch.decoder_inputs
    decoder_inputs.encoder_outputs = encoder_outputs

    hypotheses = [[target_vocabulary.SOS_TOKEN] for _ in sequences]  # Store generated predictions
    action_histories = [
        [{"symbol": symbol, "actions": [], "predictions": []} for symbol in sequence] for sequence in sequences
    ]

    sampled_tokens = [[target_vocabulary.get_symbol_index(target_vocabulary.SOS_TOKEN)] for _ in sequences]
    sampled_tokens = torch.tensor(sampled_tokens).long()
    positions = [0 for _ in sequences]

    hidden = None
    step_num = 0

    while (
            any(position < length for position, length in zip(positions, source_lengths)) and
            step_num < max_decoding_length
    ):
        step_num += 1
        positions = [min(position, source_lengths[i]-1) for i, position in enumerate(positions)]

        decoder_inputs.decoder_inputs = sampled_tokens
        decoder_inputs.decoder_input_lengths = torch.ones(len(sequences))
        decoder_inputs.hidden_states = hidden

        # Get next predicted tokens
        with torch.no_grad():
            decoder_outputs = model.decode(decoder_inputs=decoder_inputs)

            # Get predicted symbol indices
            scores = decoder_outputs.scores
            predictions = scores.argmax(dim=-1).squeeze(2)
            predictions = predictions[torch.arange(len(sequences)), positions]
            predictions = predictions.cpu().tolist()

            # Unpack Hidden states
            old_hidden = decoder_outputs.old_hidden_states
            new_hidden = decoder_outputs.new_hidden_states

        sampled_hidden = []

        for sentence_idx, prediction in enumerate(predictions):
            # If already done, ignore prediction
            if (
                    positions[sentence_idx] >= source_lengths[sentence_idx] or
                    (
                            len(hypotheses[sentence_idx]) > 0 and
                            hypotheses[sentence_idx][-1] == target_vocabulary.EOS_TOKEN
                    )
            ):
                sampled_hidden.append((new_hidden[0][:, sentence_idx], new_hidden[1][:, sentence_idx]))
                if hypotheses[sentence_idx][-1] != target_vocabulary.EOS_TOKEN:
                    hypotheses[sentence_idx].append(target_vocabulary.EOS_TOKEN)

            else:
                position = positions[sentence_idx]
                predicted_action = target_vocabulary[prediction]

                if isinstance(predicted_action, Deletion):
                    sampled_hidden.append((old_hidden[0][:, sentence_idx], old_hidden[1][:, sentence_idx]))
                    action_histories[sentence_idx][position]["actions"].append(predicted_action)
                    positions[sentence_idx] += 1

                elif isinstance(predicted_action, Substitution):
                    sampled_token = predicted_action.token
                    hypotheses[sentence_idx].append(sampled_token)
                    sampled_hidden.append((new_hidden[0][:, sentence_idx], new_hidden[1][:, sentence_idx]))
                    action_histories[sentence_idx][position]["actions"].append(predicted_action)
                    action_histories[sentence_idx][position]["predictions"].append(sampled_token)
                    positions[sentence_idx] += 1

                elif isinstance(predicted_action, Insertion):
                    sampled_token = predicted_action.token
                    hypotheses[sentence_idx].append(sampled_token)
                    sampled_hidden.append((new_hidden[0][:, sentence_idx], new_hidden[1][:, sentence_idx]))
                    action_histories[sentence_idx][position]["actions"].append(predicted_action)
                    action_histories[sentence_idx][position]["predictions"].append(sampled_token)

                elif isinstance(predicted_action, Noop):
                    sampled_hidden.append((old_hidden[0][:, sentence_idx], old_hidden[1][:, sentence_idx]))
                    action_histories[sentence_idx][position]["actions"].append(predicted_action)
                    positions[sentence_idx] += 1

                else:
                    raise RuntimeError(f"Sampled invalid action: {predicted_action}")

        h_0, c_0 = zip(*sampled_hidden)
        h_0, c_0 = torch.stack(h_0), torch.stack(c_0)
        h_0, c_0 = h_0.transpose(0, 1), c_0.transpose(0, 1)
        hidden = (h_0, c_0)
        sampled_tokens = [
            [target_vocabulary.get_symbol_index(hypothesis[-1])] for hypothesis in hypotheses
        ]
        sampled_tokens = torch.tensor(sampled_tokens).long()

    # Reformat predictions
    predictions = []
    for prediction, alignment in zip(hypotheses, action_histories):
        alignment = [
            AlignmentPosition(
                symbol=position["symbol"], actions=position["actions"], predictions=position["predictions"]
            ) for position in alignment
        ]
        predictions.append(TransducerPrediction(prediction=prediction[1:], alignment=alignment))

    return predictions


def autoregressive_beam_search_sampling(model: TransducerModel, batch: Batch, target_vocabulary: TransducerVocabulary,
                                        num_beams: int = 5,
                                        max_decoding_length: int = 70) -> List[TransducerPrediction]:
    model = model.eval()
    model = model.to(model.kwargs["device"])

    sequences = batch.raw_sources
    source_lengths = batch.source_num_segments.cpu().tolist()

    # Run encoder
    with torch.no_grad():
        encoder_outputs = model.encode(batch.encoder_inputs)

    decoder_inputs = batch.decoder_inputs
    decoder_inputs.encoder_outputs = encoder_outputs

    # Initialise beams
    beams = dict()
    sos_token = target_vocabulary.SOS_TOKEN
    eos_token = target_vocabulary.EOS_TOKEN

    for source_index, source in enumerate(sequences):
        beams[source_index] = []

        beam = Beam(
            source_index=source_index, position=0, hidden=None, predictions=[sos_token],
            alignments=[{"symbol": symbol, "actions": [], "predictions": []} for symbol in source],
            score=0.0
        )
        beams[source_index].append(beam)

    # Initialise criterion to decide whether beam is finished
    def is_finished(bm: Beam) -> bool:
        has_eos = bm.predictions[-1] == eos_token
        empty_buffer = bm.position >= len(sequences[bm.source_index])

        return has_eos or empty_buffer

    # Helper function to retrieve all beams from grouped dictionary
    def get_all_beams():
        all_beams = []
        for grouped_beams in beams.values():
            all_beams.extend(grouped_beams)

        return all_beams

    step_num = 0

    class Hypotheses:
        def __init__(self):
            self.hypotheses = [[] for _ in sequences]

        def add(self, bm: Beam, s_index: int):
            insertion_index = 0

            for stored_beam_score, stored_beam in self.hypotheses[s_index]:
                if bm.score < stored_beam_score:
                    insertion_index += 1
                else:
                    break

            if insertion_index <= num_beams:
                self.hypotheses[s_index].insert(insertion_index, (bm.score, bm))

        def get_best_score(self, s_index: int) -> float:
            if len(self.hypotheses[s_index]) == 0:
                return -torch.inf
            return self.hypotheses[s_index][0][0]

    hypotheses = Hypotheses()

    while len(get_all_beams()) > 0 and step_num < max_decoding_length:
        # Increase step counter
        step_num += 1

        # Get next predicted tokens
        with torch.no_grad():
            # Collect all active beams
            current_beams = get_all_beams()

            # Collect previously sampled symbols from active beams
            sampled_symbols = [beam.predictions[-1] for beam in current_beams]
            sampled_symbols = [[target_vocabulary.get_symbol_index(symbol)] for symbol in sampled_symbols]
            sampled_symbols = torch.tensor(sampled_symbols).long()

            # Collect source encodings of active beams
            current_source_encodings = [encoder_outputs[beam.source_index] for beam in current_beams]
            current_source_encodings = torch.stack(current_source_encodings)

            # Collect source lengths of active beams
            current_source_lengths = [source_lengths[beam.source_index] for beam in current_beams]
            current_source_lengths = torch.tensor(current_source_lengths).long().flatten()

            # Collect decoder hidden states of active beams
            if step_num > 1:
                current_hidden = [beam.hidden for beam in current_beams]
                current_h0, current_c0 = zip(*current_hidden)
                current_h0 = torch.stack(current_h0)
                current_c0 = torch.stack(current_c0)
                current_h0 = current_h0.transpose(0, 1)
                current_c0 = current_c0.transpose(0, 1)
                current_hidden = (current_h0, current_c0)
            else:
                current_hidden = None

            # Calculate new decoder hidden states
            decoder_inputs.encoder_outputs = current_source_encodings
            decoder_inputs.decoder_inputs = sampled_symbols
            decoder_inputs.decoder_input_lengths = torch.ones(len(current_beams))
            decoder_inputs.encoder_output_lengths = current_source_lengths
            decoder_inputs.hidden_states = current_hidden

            decoder_outputs = model.decode(decoder_inputs=decoder_inputs)

            # Extract Scores
            current_positions = [beam.position for beam in current_beams]
            scores = decoder_outputs.scores.argmax(dim=-1).squeeze(2).detach().cpu()
            scores = scores[torch.arange(0, len(current_beams)), current_positions]

            # Unpack Hidden States
            old_hidden = decoder_outputs.old_hidden_states
            new_hidden = decoder_outputs.new_hidden_states

        # Update beams
        new_beams = {source_index: [] for source_index in beams.keys()}

        for idx, (beam_scores, beam) in enumerate(zip(scores, current_beams)):
            branch_counter = 0
            score_rank = 0
            beam_scores = beam_scores.flatten()
            sorted_score_indices = torch.argsort(beam_scores, descending=True)

            while branch_counter < num_beams and score_rank < len(sorted_score_indices):
                predicted_index = sorted_score_indices[score_rank].item()
                score = beam_scores[predicted_index].item()

                predicted_action = target_vocabulary[predicted_index]

                if isinstance(predicted_action, Substitution) or isinstance(predicted_action, Insertion):
                    if isinstance(predicted_action, Substitution):
                        sampled_symbol = predicted_action.token
                        position_update = 1

                    else:
                        sampled_symbol = predicted_action.token
                        position_update = 0

                    # Make updated predictions
                    predictions = deepcopy(beam.predictions) + [sampled_symbol]

                    # Make updated alignment history
                    alignment = deepcopy(beam.alignments)
                    alignment[beam.position]["actions"].append(predicted_action)
                    alignment[beam.position]["predictions"].append(sampled_symbol)

                    # Get hidden
                    hidden = (new_hidden[0][:, idx], new_hidden[1][:, idx])

                    # Make updated beam
                    new_beam = Beam(
                        source_index=beam.source_index,
                        position=beam.position + position_update,
                        hidden=hidden,
                        predictions=predictions,
                        alignments=alignment,
                        score=beam.score + score
                    )

                elif predicted_action.is_deletion() or predicted_action.is_noop():
                    hidden = (old_hidden[0][:, idx], old_hidden[1][:, idx])

                    # Make updated alignment history
                    alignment = deepcopy(beam.alignments)
                    alignment[beam.position]["actions"].append(predicted_action)

                    new_beam = Beam(
                        source_index=beam.source_index,
                        position=beam.position + 1,
                        hidden=hidden,
                        predictions=deepcopy(beam.predictions),
                        alignments=alignment,
                        score=beam.score + score
                    )

                else:
                    raise RuntimeError(f"Illegal action sampled: {predicted_action}")

                if is_finished(new_beam) or step_num >= max_decoding_length:
                    hypotheses.add(bm=new_beam, s_index=new_beam.source_index)
                elif new_beam.score >= hypotheses.get_best_score(s_index=new_beam.source_index):
                    new_beams[beam.source_index].append(new_beam)
                    branch_counter += 1
                else:
                    branch_counter += 1

                score_rank += 1

        beams = {
            source_index: list(sorted(beam_candidates, key=lambda bm: -bm.score))[:num_beams]
            for source_index, beam_candidates in new_beams.items()
        }

    predictions = []
    for source_predictions in hypotheses.hypotheses:
        _, best_hypothesis = max(source_predictions, key=lambda hypothesis: hypothesis[0])

        prediction = best_hypothesis.predictions
        alignment = [
            AlignmentPosition(
                symbol=position["symbol"], predictions=position["predictions"], actions=position["actions"]
            )
            for position in best_hypothesis.alignments
        ]
        predictions.append(TransducerPrediction(prediction=prediction[1:], alignment=alignment))

    return predictions
