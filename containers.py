from collections import namedtuple
from recordclass import recordclass

Batch = namedtuple(
    "Batch",
    [
        "sources", "targets", "source_languages", "target_languages", "insertion_labels", "substitution_labels",
        "source_num_segments", "source_num_languages", "target_lengths", "deletion_index", "noop_index",
        "raw_sources", "raw_source_languages", "raw_targets", "raw_target_languages", "encoder_inputs", "decoder_inputs"
    ]
)

RawDataset = namedtuple(
    "RawDataset",
    field_names=[
        "source_cognate_sets", "source_languages", "targets", "target_languages"
    ]
)
RawBatchElement = namedtuple(
    "RawBatch",
    field_names=[
        "cognate_set", "target", "source_languages", "target_language"
    ]
)
EncoderInput = namedtuple(
    "EncoderInput",
    field_names=[
        "cognate_sets", "num_segments", "num_languages", "source_languages", "target_languages"
    ]
)
DecoderInput = recordclass(
    "DecoderInput",
    fields=[
        "decoder_inputs", "decoder_input_lengths", "encoder_outputs", "encoder_output_lengths", "hidden_states", "tau"
    ]
)
DecoderOutput = namedtuple(
    "DecoderOutput", field_names=["scores", "old_hidden_states", "new_hidden_states"]
)
TrainedModel = recordclass(
    "TrainedModel",
    fields=[
        "model", "source_vocabulary", "target_vocabulary", "feature_vocabulary", "metrics", "checkpoint", "settings"
    ]
)
Beam = namedtuple(
    "Beam",
    field_names=[
        "source_index", "position", "hidden", "predictions", "alignments", "score"
    ]
)
AlignmentPosition = namedtuple("AlignmentPosition", ["symbol", "actions", "predictions"])
TransducerPrediction = namedtuple("TransducerPrediction", ["prediction", "alignment"])