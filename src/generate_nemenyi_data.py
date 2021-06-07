import os
import sys
import pandas as pd
import numpy as np
from utils import DATASETS
from utils import MODELS
from utils import OUT_PATH


def main(logs_folder, output_folder, table_name):
    columns_dt = list(MODELS.keys())
    metrics_names = {
        "MAE": "MAE",
        "RMSE": "RMSE",
        "R2": "R2",
        "memory": "Memory (MB)",
        "time": "Time (s)"
    }

    for metric, metricp in metrics_names.items():
        print(metricp)
        agg = pd.DataFrame(
            np.zeros((len(DATASETS), len(columns_dt))),
            columns=columns_dt
        )

        for count_dataset, dataset in enumerate(DATASETS):
            print(dataset)
            for model in MODELS:
                try:
                    log_avg = pd.read_csv(f"{logs_folder}/mean_{dataset}_{model}.csv")

                    agg.loc[count_dataset, model] = log_avg.tail(1)[metric].values[0]
                except FileNotFoundError:
                    agg.loc[count_dataset, model] = float('NaN')
                print(model)
        agg.to_csv(f"{output_folder}/{table_name}_{metric}.csv", index=False)


if __name__ == '__main__':
    logs_folder = f'{OUT_PATH}/final'
    output_folder = f'{OUT_PATH}/nemenyi'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    table_name = sys.argv[1]
    main(logs_folder, output_folder, table_name)
