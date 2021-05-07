import os
import sys
import pandas as pd
import numpy as np
from utils import DATASETS
from utils import INCLUDE_STD_IN_TABLES
from utils import MODELS
from utils import OUT_PATH
from utils import SYNTH_DATA


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
        agg_mean = pd.DataFrame(
            np.zeros((len(DATASETS), len(columns_dt))),
            columns=columns_dt
        )
        agg_std = pd.DataFrame(
            np.zeros((len(DATASETS), len(columns_dt))),
            columns=columns_dt
        )

        count_dataset = 0
        for dataset in DATASETS:
            print(dataset)
            for model in MODELS:
                try:
                    log_avg = pd.read_csv(f"{OUT_PATH}/final/mean_{dataset}_{model}.csv")
                    log_std = pd.read_csv(f"{OUT_PATH}/final/std_{dataset}_{model}.csv")

                    agg_mean.loc[count_dataset, model] = log_avg.tail(1)[metric].values[0]
                    agg_std.loc[count_dataset, model] = log_std.tail(1)[metric].values[0]
                except FileNotFoundError:
                    agg_mean.loc[count_dataset, model] = float('NaN')
                    agg_std.loc[count_dataset, model] = float('NaN')
                print(model)
            count_dataset += 1

        final_table_name = '{}/results_{}_{}.tex'.format(
            output_folder, metric, table_name
        )
        with open(final_table_name, 'w') as f:
            # Preamble portion
            f.write('\\begin{table*}[!htbp]\n')
            f.write(f'\t\\caption{{{metricp} results}}\n')
            f.write(f'\t\\label{{tab_{metric}}}\n')
            f.write('\t\\centering\n')
            f.write('\t\\setlength{\\tabcolsep}{3pt}\n')
            f.write('\t\\resizebox{\\textwidth}{!}{\n')
            f.write('\t\\begin{{tabular}}{{l{0}}}\n'.format(len(MODELS) * 'r'))
            f.write('\t\t\\toprule\n')

            # Header portion
            header = '\t\tDataset & '
            header += f'{" & ".join(MODELS.keys())}'
            header = '{}\\\\\n'.format(header)
            f.write(header)
            f.write('\t\t\\midrule\n')

            # Data portion
            general_ranks = np.zeros(agg_mean.shape[1])
            real_data_ranks = np.zeros(agg_mean.shape[1])
            synth_data_ranks = np.zeros(agg_mean.shape[1])
            count_real_datasets = 0
            count_synth_datasets = 0
            n_alg = len(MODELS)
            for i, (dataset_id, dataset) in enumerate(DATASETS.items()):
                line = [dataset]
                ################################################
                # Defining average ranks for each algorithm ####
                trow = agg_mean.iloc[i, :].values
                if metric == "R2":
                    trow = -trow

                temp = np.argsort(trow)
                ranks_ = np.zeros_like(general_ranks)
                ranks_[temp] = np.asarray(
                    [r if not np.isnan(val) else n_alg
                     for r, val in zip(np.arange(n_alg) + 1, trow[temp])]
                )
                general_ranks += ranks_

                # Rankings for the synthetic datasets
                if dataset_id in SYNTH_DATA:
                    temp = np.argsort(trow)
                    ranks_ = np.zeros_like(synth_data_ranks)

                    ranks_[temp] = np.asarray(
                        [r if not np.isnan(val) else 0
                         for r, val in zip(np.arange(n_alg) + 1, trow[temp])]
                    )
                    synth_data_ranks += ranks_
                    count_synth_datasets += 1
                else:  # Rankings for the real-world datasets
                    temp = np.argsort(trow)
                    ranks_ = np.zeros_like(real_data_ranks)
                    ranks_[temp] = np.asarray(
                        [r if not np.isnan(val) else 0
                         for r, val in zip(np.arange(n_alg) + 1, trow[temp])]
                    )
                    real_data_ranks += ranks_
                    count_real_datasets += 1
                ################################################
                for j in range(n_alg):
                    # For missing results
                    if np.isnan(agg_mean.iloc[i, j]):
                        line.append('--')
                        continue

                    flag = False
                    if metric != "R2":
                        flag = j == np.nanargmin(agg_mean.iloc[i, :].values)
                    else:
                        flag = j == np.nanargmax(agg_mean.iloc[i, :].values)

                    if flag:
                        if INCLUDE_STD_IN_TABLES:
                            line.append(
                                '$\\mathbf{{{0:.4f} \\pm {1:.2f}}}$'.format(
                                    agg_mean.iloc[i, j], agg_std.iloc[i, j]
                                )
                            )
                        else:
                            line.append(
                                '$\\mathbf{{{0:.4f}}}$'.format(
                                    agg_mean.iloc[i, j]
                                )
                            )
                    else:
                        if INCLUDE_STD_IN_TABLES:
                            line.append(
                                '${0:.4f} \\pm {1:.2f}$'.format(
                                    agg_mean.iloc[i, j], agg_std.iloc[i, j]
                                )
                            )
                        else:
                            line.append(
                                '${0:.4f}$'.format(
                                    agg_mean.iloc[i, j]
                                )
                            )
                line = '\t\t{0}\\\\\n'.format(' & '.join(line))
                f.write(line)
            f.write('\t\t\\midrule\n')
            ###################################################################
            # General ranks
            general_ranks /= len(DATASETS)
            min_ = min(general_ranks)
            general_ranks = ['${:.2f}$'.format(r) if r != min_ else
                             '$\\mathbf{{{:.2f}}}$'.format(r)
                             for r in general_ranks]
            f.write('\t\t\\textbf{{Avg. rank}} & ' + ' & '.join(general_ranks) + '\\\\\n')
            if count_real_datasets > 0 and count_synth_datasets > 0:
                real_data_ranks /= count_real_datasets
                real_data_ranks[real_data_ranks == 0] = float('Nan')
                min_ = min(real_data_ranks)
                real_data_ranks = ['--' if np.isnan(r) else
                                   '${:.2f}$'.format(r) if r != min_ else
                                   '$\\mathbf{{{:.2f}}}$'.format(r)
                                   for r in real_data_ranks]
                f.write(
                    '\t\t\\textbf{{Avg. rank real}} & ' + ' & '.join(real_data_ranks) + '\\\\\n'
                )

                synth_data_ranks /= count_synth_datasets
                synth_data_ranks[synth_data_ranks == 0] = float('Nan')
                min_ = min(synth_data_ranks)
                synth_data_ranks = ['--' if np.isnan(r) else
                                    '${:.2f}$'.format(r) if r != min_ else
                                    '$\\mathbf{{{:.2f}}}$'.format(r)
                                    for r in synth_data_ranks]
                f.write(
                    '\t\t\\textbf{{Avg. rank synth.}} & ' + ' & '.join(synth_data_ranks) + '\\\\\n'
                )
            ###################################################################
            f.write('\t\t\\bottomrule\n')
            f.write('\t\\end{tabular}}\n')
            f.write('\\end{table*}\n')


if __name__ == '__main__':
    logs_folder = f'{OUT_PATH}/final'
    output_folder = f'{OUT_PATH}/tables'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    table_name = sys.argv[1]
    main(logs_folder, output_folder, table_name)
