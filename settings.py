import os
import torch

from typing import Optional
from collections import namedtuple

setting_fields = [
    # Training Settings
    "epochs",  # Number of epochs to train
    "batch",  # Batch size
    "device",  # Whether to use cuda
    "scheduler",  # Name of learning rate scheduler
    "gamma",  # Decay factor for exponential learning rate scheduler
    "verbose",  # Whether to print training progress
    "report_progress_every",  # Number of updates between loss reports
    "evaluate_every",  # Number of epochs between evaluating on dev set
    "main_metric",  # Development metric used for evaluating training progress (training loss if no dev data)
    "keep_only_best_checkpoint",  # Whether to only save the best checkpoint (according to loss / dev score)
    "min_source_frequency",  # Mask source symbols by UNK token that appear less than given frequency
    "min_target_frequency",  # Mask target symbols by UNK token that appear less than given frequency,
    "augment_shuffle",  # Whether to shuffle languages in cognate set
    "augment_mask_languages",  # Whether to mask a random number of languages
    "expected_num_masked_languages",  # Expected number of masked languages
    "early_stopping",  # Whether to stop training if eval metric does not improve for a certain number of epochs
    "early_stopping_tolerance",  # Number of epochs to wait for early stopping

    # Optimizer Settings
    "optimizer",  # Name of optimizer
    "lr",  # (Initial / Max.) learning rate
    "weight_decay",  # Weight decay factor
    "grad_clip",  # Max. absolute value of gradients (not applied if None)

    # Model Settings
    "model",  # Model type (autoregressive, non-autoregressive, soft-attention)
    "embedding_size",  # Num dimensions of embedding vectors
    "hidden_size",  # Hidden size
    "hidden_layers",  # Num of layers of encoders / decoders
    "dropout",  # Dropout probability
    "tau",  # Branching factor for non-autoregressive model
    "max_targets_per_symbol",  # Maximum number of target symbol decoded from a single input symbol
    "encoder_bridge",  # Pass summary of source sequence to autoregressive decoder

    # Loss Settings
    "noop_discount",  # Discount factor for loss incurred by blank actions (only for non-autoregressive model)
    "fast_autoregressive",  # Use fast autoregressive loss. In this case, copying the same symbol multiple times is not
                            # possible

    # Experiment Settings
    "name",  # Name of experiment
    "train_data_path",  # Path to train data
    "dev_data_path",  # Path to dev data
    "save_path",  # Where to save model checkpoints and settings

    # Inference Settings
    "beam_search",  # Whether to use beam search decoding for autoregressive transducers
    "num_beams",  # Number of beams for beam search decoding
    "max_decoding_length",  # Maximum length of decoded sequences (only autoregressive models)
]

Settings = namedtuple("Settings", field_names=setting_fields)


def save_settings(settings: Settings) -> None:
    os.makedirs(settings.save_path, exist_ok=True)
    with open(os.path.join(settings.save_path, f"{settings.name}_settings.tsv"), 'w') as ssf:
        for setting, value in settings._asdict().items():
            ssf.write(f"{setting}\t{value}\n")


def make_settings(
        model: str, name: str, save_path: str, epochs: int = 1, batch: int = 16,
        device: torch.device = torch.device('cpu'), scheduler: str = "exponential", gamma: float = 1.0,
        verbose: bool = True, report_progress_every: int = 10, main_metric: str = "loss",
        keep_only_best_checkpoint: bool = True, optimizer: str = "sgd", lr: float = 0.001, weight_decay: float = 0.0,
        grad_clip: Optional[float] = None, embedding_size: int = 128, hidden_size: int = 128, hidden_layers: int = 1,
        dropout: float = 0.0, tau: Optional[int] = 5, evaluate_every: int = 1, max_targets_per_symbol: int = 50,
        noop_discount: float = 1.0,  train_data_path: Optional[str] = None,
        dev_data_path: Optional[str] = None, beam_search: bool = True, num_beams: int = 5,
        max_decoding_length: int = 100, encoder_bridge: bool = False, min_source_frequency: int = 1,
        min_target_frequency: int = 1, fast_autoregressive: bool = False, augment_shuffle: bool = False,
        augment_mask_languages: bool = False, expected_num_masked_languages: int = 1, early_stopping: bool = False,
        early_stopping_tolerance: int = 3) -> Settings:
    return Settings(
        epochs=epochs, batch=batch, device=device, scheduler=scheduler, gamma=gamma,
        verbose=verbose, report_progress_every=report_progress_every, main_metric=main_metric,
        keep_only_best_checkpoint=keep_only_best_checkpoint,
        optimizer=optimizer, lr=lr, weight_decay=weight_decay, grad_clip=grad_clip, model=model,
        embedding_size=embedding_size, hidden_size=hidden_size, hidden_layers=hidden_layers, dropout=dropout,
        tau=tau, max_targets_per_symbol=max_targets_per_symbol, noop_discount=noop_discount,  name=name,
        train_data_path=train_data_path, dev_data_path=dev_data_path, save_path=save_path, beam_search=beam_search,
        num_beams=num_beams, max_decoding_length=max_decoding_length, encoder_bridge=encoder_bridge,
        evaluate_every=evaluate_every, fast_autoregressive=fast_autoregressive,
        min_source_frequency=min_source_frequency, min_target_frequency=min_target_frequency,
        augment_shuffle=augment_shuffle, augment_mask_languages=augment_mask_languages,
        expected_num_masked_languages=expected_num_masked_languages, early_stopping=early_stopping,
        early_stopping_tolerance=early_stopping_tolerance
    )
