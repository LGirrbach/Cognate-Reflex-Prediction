import sys
import torch
import optuna
import logging
import numpy as np
import editdistance

from typing import List
from transducer import Transducer
from settings import make_settings
from cognate_prediction_test import read_test_dataset
from cognate_prediction_test import read_train_dataset


def get_settings(model: str, num_layers: int, hidden_size: int, dropout: float, batch_size: int, lr: float,
                 gamma: float, epochs: int):
    name = f"cognate_tuning-model={model}-num_layers={num_layers}-hidden_size={hidden_size}"
    name = name + f"-dropout={dropout}-batch_size={batch_size}-lr={lr}-gamma={gamma}-epochs={epochs}"
    settings = make_settings(
        model=model, name=name, save_path="./results/tuning/saved_models",
        epochs=epochs, device=torch.device("cuda:0"), optimizer="adam", batch=batch_size, tau=5, verbose=False,
        noop_discount=10.0, hidden_layers=num_layers, scheduler="exponential", gamma=gamma, lr=lr,
        beam_search=False, num_beams=5, grad_clip=1.0, evaluate_every=5, fast_autoregressive=True,
        augment_shuffle=False, augment_mask_languages=False, dropout=dropout, hidden_size=hidden_size
    )

    return settings


def evaluate(y_true: List[List[str]], y_pred: List[List[str]]):
    wer = 100 * (1 - np.mean([prediction == target for prediction, target in zip(y_pred, y_true)]))
    edit_distance = np.mean(
        [editdistance.distance(prediction, target) for prediction, target in zip(y_pred, y_true)]
    )

    return {"wer": wer, "edit-distance": edit_distance}


if __name__ == '__main__':
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    data_name = "hillburmish"
    train_data = read_train_dataset(data_name, level=10)
    test_cognate_sets, test_languages_source, test_languages_target = read_test_dataset(data_name, level=10)

    with open(f"data/{data_name}/solutions-0.10.tsv") as tf:
        test_targets = []
        for line in tf:
            entries = line.strip().split("\t")
            entries = [entry.strip() for entry in entries if entry.strip()]
            if len(entries) == 2:
                test_targets.append(entries[1].split())

    assert len(test_targets) == len(test_cognate_sets)

    model_options = (
        "autoregressive", "non-autoregressive-lstm", "non-autoregressive-position", "non-autoregressive-fixed"
    )

    for model_name in model_options:
        def objective(trial: optuna.Trial):
            num_layers = trial.suggest_categorical("num_layers", [1, 2])
            hidden_size = trial.suggest_int("hidden_size", 64, 512)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            batch_size = trial.suggest_int("batch_size", 4, 32)
            lr = trial.suggest_float("lr", 0.0005, 0.01, log=True)
            gamma = trial.suggest_float("gamma", 0.9, 1.0)
            epochs = trial.suggest_int("epochs", 10, 40)

            settings = get_settings(
                model=model_name, num_layers=num_layers, hidden_size=hidden_size, dropout=dropout,
                batch_size=batch_size,  lr=lr, gamma=gamma, epochs=epochs
            )
            transducer = Transducer(settings=settings)
            transducer.fit(train_data=train_data, development_data=None)

            predictions = transducer.predict(
                sources=test_cognate_sets, source_languages=test_languages_source,
                target_languages=test_languages_target
            )
            predictions = [prediction.prediction for prediction in predictions]
            predictions = [
                [token for token in prediction if not (token.startswith("<") and token.endswith(">"))]
                for prediction in predictions
            ]

            scores = evaluate(y_true=test_targets, y_pred=predictions)
            return scores["edit-distance"]


        study_name = f"cognates-tuning-model={model_name}"
        storage_name = f"sqlite:///results/tuning/{study_name}.db"
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize")
        study.optimize(objective, n_trials=50)

        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        df.to_csv(f"results/tuning/{study_name}.csv")
