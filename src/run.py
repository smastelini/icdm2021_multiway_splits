import copy
import numbers
import random

import pandas as pd

from river import compose
from river import evaluate
from river import metrics
from river import preprocessing
from river import stream
from river import synth

from utils import CAT_FEATURES, DATASETS, GENERATOR_BASED, IN_PATH, MAIN_SEED, MODELS, N_REPS, OUT_PATH


def prepare_data(dataset_name, seed=None):
    if dataset_name not in GENERATOR_BASED:
        header = ""
        with open(f"{IN_PATH}/{dataset_name}.csv", "r") as f:
            header = f.readline()

        names = header.replace("\n", "").split(",")
        features, target = names[:-1], names[-1]

        if dataset_name in CAT_FEATURES:
            nominal_attributes = [features[i] for i in CAT_FEATURES[dataset_name]]
        else:
            nominal_attributes = None

        dataset = stream.iter_csv(
            f"{IN_PATH}/{dataset_name}.csv",
            target=target,
            converters={
                name: float for name in names
            }
        )

        dataset = stream.shuffle(
            dataset,
            buffer_size=100,
            seed=seed
        )
    else:
        nominal_attributes = None
        if dataset_name == "friedman":
            dataset = synth.Friedman(seed=seed).take(100_000)
        elif dataset_name == "planes2d":
            dataset = synth.Planes2D(seed=seed).take(100_000)
        elif dataset_name == "mv":
            dataset = synth.Mv(seed=seed).take(100_000)

    return dataset, nominal_attributes


def run_dataset(dataset_name):
    print(dataset_name)
    rng = random.Random(MAIN_SEED)
    seeds = [rng.randint(0, 99999) for _ in range(N_REPS)]
    for model_name, template_model in MODELS.items():
        print(model_name, end="\t")
        summaries = {}
        for rep in range(N_REPS):
            print(f" {rep}", end="")
            dataset, nominal_attributes = prepare_data(dataset_name, seed=seeds[rep])
            model = copy.deepcopy(template_model)

            if dataset_name in CAT_FEATURES:
                model._set_params({"nominal_attributes": CAT_FEATURES[dataset_name]})
                preproc = (
                    (compose.Discard(*tuple(nominal_attributes)) | compose.SelectType(
                        numbers.Number) | preprocessing.StandardScaler()) + compose.Select(
                        *tuple(nominal_attributes)) + compose.SelectType(str)
                )
            else:
                preproc = (
                    (
                        compose.SelectType(numbers.Number) | preprocessing.StandardScaler()
                    ) + compose.SelectType(str)
                )
            model = preproc | model

            evaluate.progressive_val_score(
                dataset,
                model,
                metrics.MAE() + metrics.RMSE() + metrics.R2(),
                show_memory=True,
                show_time=True,
                print_every=100,
                file=open(
                    f"{OUT_PATH}/results_{dataset_name}_{model_name}_rep{rep:02d}.txt",
                    "w"
                )
            )

            summaries[rep] = model["HoeffdingTreeRegressor"].summary
        print()

        pd.DataFrame.from_dict(summaries, orient="index").to_csv(
            f"{OUT_PATH}/stats_{dataset_name}_{model_name}.txt"
        )


for dataset_name in DATASETS:
    run_dataset(dataset_name)
