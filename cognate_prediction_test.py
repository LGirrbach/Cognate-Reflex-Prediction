import os
import torch
import pandas as pd

from lingpy import Multiple
from transducer import Transducer
from containers import RawDataset
from settings import make_settings

base_path = "."
datasets = list(sorted(os.listdir(f"{base_path}/data")))


def read_train_dataset(dataset_name: str, level: int):
    assert dataset_name in datasets
    assert level in [10, 20, 30, 40, 50]

    train_dataset = pd.read_csv(os.path.join(base_path, "data", dataset_name, f"training-0.{level}.tsv"), sep="\t")
    train_dataset = train_dataset.drop("COGID", axis=1)

    languages = list(train_dataset.columns)
    train_cognate_sets = []
    train_languages = []

    for _, cognate_set in train_dataset.iterrows():
        current_cognate_set = []
        current_languages = []

        for cognate, language in zip(cognate_set.tolist(), languages):
            if isinstance(cognate, float):
                cognate = ""

            cognate = cognate.strip().split()

            if len(cognate) > 0:
                current_cognate_set.append(cognate)
                current_languages.append(language)

        if len(current_cognate_set) > 1:
            train_cognate_sets.append(current_cognate_set)
            train_languages.append(current_languages)

    expanded_train_cognate_sets = []
    expanded_train_source_languages = []
    train_targets = []
    target_languages = []

    for cognate_set, cognate_set_languages in zip(train_cognate_sets, train_languages):
        for target_idx, (target, target_language) in enumerate(zip(cognate_set, cognate_set_languages)):
            source_cognate_set = [cognate for i, cognate in enumerate(cognate_set) if i != target_idx]
            source_languages = [language for i, language in enumerate(cognate_set_languages) if i != target_idx]

            alignment = Multiple(source_cognate_set)
            alignment.prog_align()
            alignment = alignment.alm_matrix

            expanded_train_cognate_sets.append(alignment)
            expanded_train_source_languages.append(source_languages)
            train_targets.append(target)
            target_languages.append(target_language)

    return RawDataset(
        source_cognate_sets=expanded_train_cognate_sets, source_languages=expanded_train_source_languages,
        targets=train_targets, target_languages=target_languages
    )


def read_test_dataset(dataset_name: str, level: int):
    assert dataset_name in datasets
    assert level in [10, 20, 30, 40, 50]

    train_dataset = pd.read_csv(os.path.join(base_path, "data", dataset_name, f"test-0.{level}.tsv"), sep="\t")
    train_dataset = train_dataset.drop("COGID", axis=1)

    languages = list(train_dataset.columns)
    test_source_cognate_sets = []
    test_source_languages = []
    test_target_languages = []

    for _, cognate_set in train_dataset.iterrows():
        current_cognate_set = []
        current_languages = []

        for cognate, language in zip(cognate_set.tolist(), languages):
            if isinstance(cognate, float):
                cognate = ""

            if cognate == "?":
                test_target_languages.append(language)
                continue

            cognate = cognate.strip().split()

            if len(cognate) > 0:
                current_cognate_set.append(cognate)
                current_languages.append(language)

        if len(current_cognate_set) > 0:
            alignment = Multiple(current_cognate_set)
            alignment.prog_align()
            alignment = alignment.alm_matrix

            test_source_cognate_sets.append(alignment)
            test_source_languages.append(current_languages)

    assert len(test_source_cognate_sets) == len(test_source_languages) == len(test_target_languages)
    return test_source_cognate_sets, test_source_languages, test_target_languages


if __name__ == '__main__':
    data_name = "hattorijaponic"
    train_data = read_train_dataset(data_name, level=10)

    settings = make_settings(
        model="autoregressive", name="cognate_prediction", save_path=f"{base_path}/saved_models/test",
        epochs=20, device=torch.device("cuda:0"), optimizer="adam",  batch=16, tau=10, report_progress_every=20,
        noop_discount=10.0, hidden_layers=1, scheduler="exponential", gamma=0.99, lr=0.001,
        beam_search=False, num_beams=5, grad_clip=1.0, evaluate_every=5, fast_autoregressive=True,
        augment_shuffle=True, augment_mask_languages=True
    )

    model = Transducer(settings=settings)
    model = model.fit(train_data=train_data, development_data=None)
    # model = Transducer.load(f"{base_path}/saved_models/test/cognate_prediction.pt")

    test_cognate_sets, test_languages_source, test_languages_target = read_test_dataset(data_name, level=10)
    predictions = model.predict(
        sources=test_cognate_sets, source_languages=test_languages_source, target_languages=test_languages_target
    )

    k = 25

    print(predictions[k])
    print(test_cognate_sets[k])
