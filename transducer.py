from __future__ import annotations

from tqdm import tqdm
from typing import List
from trainer import train
from typing import Optional
from containers import Batch
from settings import Settings
from dataset import RawDataset
from trainer import load_model
from trainer import TrainedModel
from align import autoregressive_align
from torch.utils.data import DataLoader
from align import non_autoregressive_align
from containers import TransducerPrediction
from inference import non_autoregressive_inference
from inference import autoregressive_greedy_sampling
from dataset import AutoregressiveTransducerDatasetTrain
from inference import autoregressive_beam_search_sampling
from dataset import NonAutoregressiveTransducerDatasetTrain

Sequences = List[List[str]]


class Transducer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Optional[TrainedModel] = None

    @classmethod
    def load(cls, path: str) -> Transducer:
        model = load_model(path=path)
        transducer = cls(settings=model.settings)
        transducer.model = model

        return transducer

    def fit(self, train_data: RawDataset, development_data: Optional[RawDataset] = None) -> Transducer:
        self.model = train(train_data=train_data, development_data=development_data, settings=self.settings)
        return self

    def predict(self, sources: List[List[List[str]]], source_languages: List[List[str]],
                target_languages: List[str]) -> List[TransducerPrediction]:
        if self.model is None:
            raise RuntimeError("Running inference with uninitialised model")

        autoregressive = not self.settings.model.startswith("non-autoregressive")
        dummy_targets = [[] for _ in sources]
        dataset = RawDataset(
            source_cognate_sets=sources, source_languages=source_languages, targets=dummy_targets,
            target_languages=target_languages
        )
        if autoregressive:
            dataset_class = AutoregressiveTransducerDatasetTrain
        else:
            dataset_class = NonAutoregressiveTransducerDatasetTrain

        dataset = dataset_class(
            dataset=dataset, source_vocabulary=self.model.source_vocabulary,
            target_vocabulary=self.model.target_vocabulary, feature_vocabulary=self.model.feature_vocabulary,
            augment_shuffle=False, augment_mask_languages=False
        )
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.settings.batch, shuffle=False, collate_fn=dataset.collate_fn
        )

        predictions = []
        if self.settings.verbose:
            dataloader = tqdm(dataloader, desc="Prediction Progress")

        for batch in dataloader:
            predictions.extend(self._predict_batch(batch))

        return predictions

    def _predict_autoregressive_batch(self, batch: Batch) -> List[TransducerPrediction]:
        if self.settings.beam_search:
            return autoregressive_beam_search_sampling(
                model=self.model.model, batch=batch, target_vocabulary=self.model.target_vocabulary,
                num_beams=self.settings.num_beams, max_decoding_length=self.settings.max_decoding_length
            )
        else:
            return autoregressive_greedy_sampling(
                model=self.model.model, batch=batch, target_vocabulary=self.model.target_vocabulary,
                max_decoding_length=self.settings.max_decoding_length
            )

    def _predict_non_autoregressive_batch(self, batch: Batch) -> List[TransducerPrediction]:
        return non_autoregressive_inference(
            model=self.model.model, batch=batch, target_vocabulary=self.model.target_vocabulary,
            max_predictions_per_symbol=self.settings.max_targets_per_symbol
        )

    def _predict_batch(self, batch: Batch) -> List[TransducerPrediction]:
        if self.settings.model == "autoregressive":
            return self._predict_autoregressive_batch(batch=batch)
        else:
            return self._predict_non_autoregressive_batch(batch=batch)

    def align(self, sources: Sequences, targets: Sequences, features: Optional[Sequences] = None):
        if self.settings.model == "soft-attention":
            raise RuntimeError("Can't use soft attention model as aligner")
        elif self.settings.model == "autoregressive":
            return autoregressive_align(
                settings=self.settings, model=self.model.model, source_vocabulary=self.model.source_vocabulary,
                target_vocabulary=self.model.target_vocabulary, sources=sources, targets=targets,
                feature_vocabulary=self.model.feature_vocabulary, features=features
            )
        elif self.settings.model == "non-autoregressive":
            return non_autoregressive_align(
                settings=self.settings, model=self.model.model, source_vocabulary=self.model.source_vocabulary,
                target_vocabulary=self.model.target_vocabulary, sources=sources, targets=targets,
                feature_vocabulary=self.model.feature_vocabulary, features=features
            )
