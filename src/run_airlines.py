import os
import copy
import math
import numbers
import sys
import time

import pandas as pd

from river import compose
from river import metrics
from river import stream
from river import preprocessing

from utils import CAT_FEATURES, IN_PATH, MODELS, OUT_PATH

CHECK_EVERY = 1000

if not os.path.exists(f"{OUT_PATH}/airlines_study_case"):
    os.makedirs(f"{OUT_PATH}/airlines_study_case")


def prepare():
    header = ""
    with open(f"{IN_PATH}/airlines07_08.csv", "r") as f:
        header = f.readline()

    names = header.replace("\n", "").split(",")
    features, target = names[:-1], names[-1]

    nominal_attributes = [features[i] for i in CAT_FEATURES["airlines07_08"]]

    dataset = stream.iter_csv(
        f"{IN_PATH}/airlines07_08.csv",
        target=target,
        converters={
            name: float for name in names
        }
    )

    return dataset, nominal_attributes


def run(model_name):
    model = copy.deepcopy(MODELS[model_name])
    dataset, nominal_attributes = prepare()

    model.nominal_attributes = nominal_attributes
    model.max_size = math.inf
    model.memory_estimate_period = math.inf

    preproc = (
        (compose.Discard(*tuple(nominal_attributes)) | compose.SelectType(
            numbers.Number) | preprocessing.StandardScaler()) + compose.Select(
            *tuple(nominal_attributes))
    )

    model = preproc | model

    mae = metrics.MAE()
    rmse = metrics.RMSE()
    r2 = metrics.R2()

    log = {}
    index = 0

    learn_time = 0
    pred_time = 0
    for i, (x, y) in enumerate(dataset):
        start = time.time()
        y_pred = model.predict_one(x)
        pred_time += (time.time() - start)

        mae.update(y, y_pred)
        rmse.update(y, y_pred)
        r2.update(y, y_pred)

        start = time.time()
        model.learn_one(x, y)
        learn_time += (time.time() - start)

        if i > 0 and (i + 1) % CHECK_EVERY == 0:
            mem = model["HoeffdingTreeRegressor"]._raw_memory_usage / (2 ** 20)

            tmp = {
                "n_samples": i,
                "MAE": mae.get(),
                "RMSE": rmse.get(),
                "R2": r2.get(),
                "time_learning": learn_time,
                "time_predicting": pred_time,
                "memory": mem
            }

            tmp.update(model["HoeffdingTreeRegressor"].summary)

            log[index] = tmp

            pd.DataFrame.from_dict(
                log, orient="index"
            ).to_csv(
                f"{OUT_PATH}/airlines_study_case/{model_name}.csv"
            )

            index += 1


if __name__ == "__main__":
    model_name = sys.argv[1]

    run(model_name)
