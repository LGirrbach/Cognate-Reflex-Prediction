import torch
import random
import numpy as np

from typing import List
from typing import Iterable
from containers import Batch
from scipy.stats import poisson
from containers import RawDataset
from torch.utils.data import Dataset
from containers import RawBatchElement
from vocabulary import SourceVocabulary
from torch.nn.utils.rnn import pad_sequence
from vocabulary import TransducerVocabulary

from containers import EncoderInput
from containers import DecoderInput


class TransducerDatasetTrain(Dataset):
    def __init__(self, dataset: RawDataset, source_vocabulary: SourceVocabulary,
                 target_vocabulary: TransducerVocabulary, feature_vocabulary: SourceVocabulary,
                 augment_shuffle: bool = False, augment_mask_languages: bool = False,
                 augment_expected_num_masked_languages: int = 2):
        super(TransducerDatasetTrain, self).__init__()

        self.augment_shuffle = augment_shuffle
        self.augment_mask_languages = augment_mask_languages
        self.augment_expected_num_masked_languages = augment_expected_num_masked_languages

        self.source_cognate_sets = dataset.source_cognate_sets
        self.source_languages = dataset.source_languages
        self.targets = dataset.targets
        self.target_languages = dataset.target_languages

        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary
        self.feature_vocabulary = feature_vocabulary

        self.target_deletion_index = self.target_vocabulary.get_deletion_index()
        self.target_noop_index = self.target_vocabulary.get_noop_index()

    def __len__(self) -> int:
        return len(self.source_cognate_sets)

    def __getitem__(self, idx: int) -> RawBatchElement:
        # Get cognate set and according languages from stored data
        cognate_set = self.source_cognate_sets[idx]
        source_languages = self.source_languages[idx]
        target = self.targets[idx]
        target_language = self.target_languages[idx]

        # Apply data augmentation
        cognate_set, source_languages = self._augment(cognate_set=cognate_set, languages=source_languages)

        # Remove empty columns
        num_segments = len(cognate_set[0])
        all_gap_indices = [j for j in range(num_segments) if all([cognate[j] == "-" for cognate in cognate_set])]
        all_gap_indices = set(all_gap_indices)
        cognate_set = [
            [cognate[j] for j in range(num_segments) if j not in all_gap_indices] for cognate in cognate_set
        ]

        return RawBatchElement(
            cognate_set=cognate_set, target=target, source_languages=source_languages, target_language=target_language
        )

    def _augment(self, cognate_set: List[List[str]], languages: List[str]):
        if self.augment_mask_languages:
            # Sample number of masked languages from poisson distribution
            num_masked_languages = poisson.rvs(mu=self.augment_expected_num_masked_languages)
            num_masked_languages = min(num_masked_languages, len(cognate_set) - 1)

            # Compute updated indices
            cognate_set_indices = list(range(len(cognate_set)))
            masked_language_indices = random.sample(cognate_set_indices, k=num_masked_languages)
            cognate_set_indices = [idx for idx in cognate_set_indices if idx not in masked_language_indices]

            # Update cognate set and languages
            cognate_set = [cognate_set[idx] for idx in cognate_set_indices]
            languages = [languages[idx] for idx in cognate_set_indices]

        if self.augment_shuffle:
            # Shuffle indices
            cognate_set_indices = list(range(len(cognate_set)))
            random.shuffle(cognate_set_indices)

            # Update cognate set and languages
            cognate_set = [cognate_set[idx] for idx in cognate_set_indices]
            languages = [languages[idx] for idx in cognate_set_indices]

        return cognate_set, languages

    def _add_special_tokens_source(self, sequence: List[str]) -> List[str]:
        raise NotImplementedError

    def _add_special_tokens_target(self, sequence: List[str]) -> List[str]:
        raise NotImplementedError

    def _collate_sources(self, sources: List[List[List[str]]]):
        """"We assume all sources in a cognate set are aligned, i.e. have same length"""
        indexed_sources = []
        raw_sources = []
        num_segments = []
        cognate_set_sizes = []

        for cognate_set_sources in sources:
            cognate_set_sources = [self._add_special_tokens_source(source) for source in cognate_set_sources]
            raw_sources.append(np.array(cognate_set_sources).T.tolist())

            assert all([len(source) == len(cognate_set_sources[0]) for source in cognate_set_sources])
            cognate_set_num_segments = len(cognate_set_sources[0])
            cognate_set_size = len(cognate_set_sources)

            indexed_cognate_set_sources = [
                self.source_vocabulary.index_sequence(source) for source in cognate_set_sources
            ]
            indexed_cognate_set_sources = torch.tensor(indexed_cognate_set_sources).long()

            indexed_sources.append(indexed_cognate_set_sources)
            num_segments.append(cognate_set_num_segments)
            cognate_set_sizes.append(cognate_set_size)

        num_segments = torch.tensor(num_segments).long()
        cognate_set_sizes = torch.tensor(cognate_set_sizes).long()

        max_num_segments = torch.max(num_segments).long().item()
        max_cognate_set_size = torch.max(cognate_set_sizes).long().item()

        padded_template = torch.zeros(max_cognate_set_size, max_num_segments, dtype=torch.long)
        padded_indexed_cognate_set_sources = []

        for indexed_cognate_set_source in indexed_sources:
            padded_indexed_cognate_set_source = torch.clone(padded_template)
            current_num_cognate_sets, current_num_segments = indexed_cognate_set_source.shape
            padded_indexed_cognate_set_source[:current_num_cognate_sets, :current_num_segments] =\
                indexed_cognate_set_source
            padded_indexed_cognate_set_sources.append(padded_indexed_cognate_set_source)

        padded_indexed_cognate_set_sources = torch.stack(padded_indexed_cognate_set_sources)

        return {
            "sources": padded_indexed_cognate_set_sources,
            "num_segments": num_segments,
            "cognate_set_sizes": cognate_set_sizes,
            "raw_sources": raw_sources
        }

    def _collate_targets(self, targets: List[List[str]]):
        # Add special tokens
        targets = [self._add_special_tokens_target(target) for target in targets]
        # Get target lengths
        target_lengths = torch.tensor([len(sequence) for sequence in targets]).long()

        # Index targets
        indexed_targets = [
            torch.tensor(self.target_vocabulary.index_sequence(sequence)).long() for sequence in targets
        ]
        indexed_targets = pad_sequence(indexed_targets, padding_value=0, batch_first=True)

        # Get substitution indices
        substitution_labels = [
            [self.target_vocabulary.get_substitution_index(symbol) for symbol in target] for target in targets
        ]
        substitution_labels = [torch.tensor(labels).long() for labels in substitution_labels]
        substitution_labels = pad_sequence(substitution_labels, padding_value=0, batch_first=True)

        # Get insertion labels
        insertion_labels = [
            [self.target_vocabulary.get_insertion_index(symbol) for symbol in target] for target in targets
        ]
        insertion_labels = [torch.tensor(labels).long() for labels in insertion_labels]
        insertion_labels = pad_sequence(insertion_labels, padding_value=0, batch_first=True)

        return {
            'targets': indexed_targets,
            'target_lengths': target_lengths,
            'substitution_labels': substitution_labels,
            'insertion_labels': insertion_labels,
            "raw_targets": targets
        }

    def collate_fn(self, batch: Iterable[RawBatchElement]) -> Batch:
        # Unpack Batch
        cognate_sets = [batch_element.cognate_set for batch_element in batch]
        source_languages = [batch_element.source_languages for batch_element in batch]
        targets = [batch_element.target for batch_element in batch]
        target_languages = [batch_element.target_language for batch_element in batch]

        collated_sources = self._collate_sources(sources=cognate_sets)
        collated_targets = self._collate_targets(targets=targets)

        indexed_source_languages = [
            self.feature_vocabulary.index_sequence(languages) for languages in source_languages
        ]
        indexed_source_languages = [torch.tensor(languages).long() for languages in indexed_source_languages]
        indexed_source_languages = pad_sequence(indexed_source_languages, padding_value=0, batch_first=True)

        indexed_target_languages = self.feature_vocabulary.index_sequence(target_languages)
        indexed_target_languages = torch.tensor(indexed_target_languages).long()

        encoder_inputs = EncoderInput(
            cognate_sets=collated_sources["sources"], num_segments=collated_sources["num_segments"],
            num_languages=collated_sources["cognate_set_sizes"], source_languages=indexed_source_languages,
            target_languages=indexed_target_languages
        )

        decoder_inputs = DecoderInput(
            decoder_inputs=collated_targets["targets"], decoder_input_lengths=collated_targets["target_lengths"],
            encoder_outputs=None, encoder_output_lengths=collated_sources["num_segments"], hidden_states=None,
            tau=None
        )

        return Batch(
            sources=collated_sources["sources"],
            targets=collated_targets["targets"],
            source_languages=indexed_source_languages,
            target_languages=indexed_target_languages,
            insertion_labels=collated_targets["insertion_labels"],
            substitution_labels=collated_targets["substitution_labels"],
            source_num_segments=collated_sources["num_segments"],
            source_num_languages=collated_sources["cognate_set_sizes"],
            target_lengths=collated_targets["target_lengths"],
            deletion_index=self.target_deletion_index,
            noop_index=self.target_noop_index,
            raw_sources=collated_sources["raw_sources"],
            raw_source_languages=source_languages,
            raw_targets=collated_targets["raw_targets"],
            raw_target_languages=target_languages,
            encoder_inputs=encoder_inputs,
            decoder_inputs=decoder_inputs
        )


class AutoregressiveTransducerDatasetTrain(TransducerDatasetTrain):
    def _add_special_tokens_source(self, sequence: List[str]) -> List[str]:
        sequence = [self.source_vocabulary.SOS_TOKEN] + sequence + [self.source_vocabulary.EOS_TOKEN]
        return sequence

    def _add_special_tokens_target(self, sequence: List[str]) -> List[str]:
        sequence = [self.target_vocabulary.SOS_TOKEN, self.target_vocabulary.SOS_TOKEN] + sequence
        sequence = sequence + [self.target_vocabulary.EOS_TOKEN]
        return sequence


class NonAutoregressiveTransducerDatasetTrain(TransducerDatasetTrain):
    def _add_special_tokens_source(self, sequence: List[str]) -> List[str]:
        sequence = [self.source_vocabulary.SOS_TOKEN] + sequence + [self.source_vocabulary.EOS_TOKEN]
        return sequence

    def _add_special_tokens_target(self, sequence: List[str]) -> List[str]:
        sequence = [self.target_vocabulary.SOS_TOKEN] + sequence + [self.target_vocabulary.EOS_TOKEN]
        return sequence
