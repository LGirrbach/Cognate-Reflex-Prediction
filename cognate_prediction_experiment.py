import os
import torch
import pickle
import pandas as pd

from tqdm.auto import tqdm
from itertools import product
from transducer import Transducer
from settings import make_settings
from cognate_prediction_test import base_path
from cognate_prediction_test import read_test_dataset
from cognate_prediction_test import read_train_dataset


datasets = [
    "felekesemitic", "bantubvd", "listsamplesize", "kesslersignificance", "hattorijaponic", "mannburmish",
    "beidazihui", "luangthongkumkaren", "birchallchapacuran", "davletshinaztecan", "bodtkhobwa"
]


def get_model_options():
    model_names = (
        "autoregressive", "non-autoregressive-lstm", "non-autoregressive-position", "non-autoregressive-fixed"
    )

    yield from iter(model_names)


def get_hyperparameters():
    hyperparameters = dict()

    for model in get_model_options():
        tuning_file_path = os.path.join(base_path, f"results/tuning/cognates-tuning-model={model}.csv")
        tuning_data = pd.read_csv(tuning_file_path)
        model_best_hyperparameters = tuning_data[tuning_data["value"] == tuning_data["value"].min()]

        parameter_names = [
            column_name for column_name in list(tuning_data.columns) if column_name.startswith("params_")
        ]

        model_hyperparameters = {
            "_".join(parameter_name.split('_')[1:]): model_best_hyperparameters[parameter_name].item()
            for parameter_name in parameter_names
        }

        hyperparameters[model] = model_hyperparameters

    return hyperparameters


def get_settings(trial_num: int, model: str, num_layers: int, hidden_size: int, dropout: float, batch_size: int,
                 lr: float, gamma: float, epochs: int, shuffle: bool, mask_languages: bool):
    name = f"cognate_tuning-model={model}-num_layers={num_layers}-hidden_size={hidden_size}"
    name = name + f"-dropout={dropout}-batch_size={batch_size}-lr={lr}-gamma={gamma}-epochs={epochs}"
    name = name + f"-augment_shuffle={shuffle}-augment_mask_languages={mask_languages}"
    name = name + f"-trial={trial_num}"

    return make_settings(
        model=model, name=name, save_path=f"{base_path}/results/evaluation/saved_models",
        epochs=epochs, device=torch.device("cuda:0"), optimizer="adam", batch=batch_size, tau=5, verbose=False,
        noop_discount=10.0, hidden_layers=num_layers, scheduler="exponential", gamma=gamma, lr=lr,
        beam_search=False, num_beams=5, grad_clip=1.0, evaluate_every=5, fast_autoregressive=True,
        augment_shuffle=augment_shuffle, augment_mask_languages=augment_mask_languages, dropout=dropout,
        hidden_size=hidden_size
    )


if __name__ == '__main__':
    best_hyperparameters = get_hyperparameters()
    prediction_save_path = os.path.join(base_path, "results/evaluation/predictions")
    os.makedirs(prediction_save_path, exist_ok=True)

    pbar = tqdm(desc="Progress", total=len(datasets) * 4 * 4 * 5)

    for dataset in datasets:
        train_data = read_train_dataset(dataset, level=10)
        test_cognate_sets, test_languages_source, test_languages_target = read_test_dataset(dataset, level=10)

        with open(f"{base_path}/data/{dataset}/solutions-0.10.tsv") as tf:
            test_targets = []
            for line in tf:
                entries = line.strip().split("\t")
                entries = [entry.strip() for entry in entries if entry.strip()]
                if len(entries) == 2:
                    test_targets.append(entries[1].split())

        assert len(test_targets) == len(test_cognate_sets)

        for model_name in get_model_options():
            for augment_shuffle, augment_mask_languages in product([False, True], [False, True]):
                for trial in range(1, 5+1):
                    settings = get_settings(
                        trial_num=trial, model=model_name, shuffle=augment_shuffle,
                        mask_languages=augment_mask_languages, **best_hyperparameters[model_name]
                    )
                    transducer = Transducer(settings=settings)
                    transducer.fit(train_data=train_data, development_data=None)

                    predictions = transducer.predict(
                        sources=test_cognate_sets, source_languages=test_languages_source,
                        target_languages=test_languages_target
                    )

                    prediction_file_name = f"cognates-model={model_name}-dataset={dataset}"
                    prediction_file_name = prediction_file_name + f"-augment_shuffle={augment_shuffle}"
                    prediction_file_name = prediction_file_name + f"-augment_mask_languages={augment_mask_languages}"
                    prediction_file_name = prediction_file_name + f"-trial={trial}.pickle"
                    with open(os.path.join(prediction_save_path, prediction_file_name), "wb") as rf:
                        pickle.dump([predictions, test_targets], rf)

                    pbar.update(1)

    pbar.close()
