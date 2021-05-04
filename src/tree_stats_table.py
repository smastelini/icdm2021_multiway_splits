import os
import pandas as pd
import numpy as np
from utils import DATASETS
from utils import INCLUDE_STD_IN_TABLES
from utils import MODELS
from utils import OUT_PATH
from utils import SYNTH_DATA


def main(output_folder):
    columns_dt = list(MODELS.keys())
    metrics_names = {
        "n_nodes": "\# Nodes",
        "n_branches": "\# Decision nodes",
        "n_leaves": "\# Leaves",
        "height": "Height"
    }

    metric_ids = list(metrics_names.keys())
    pmetrics = list(metrics_names.values())
    agg_mean = pd.DataFrame(
        np.zeros((len(metrics_names) * len(DATASETS), len(columns_dt))),
        columns=columns_dt
    )
    agg_std = pd.DataFrame(
        np.zeros((len(metrics_names) * len(DATASETS), len(columns_dt))),
        columns=columns_dt
    )

    for i, dataset in enumerate(DATASETS):
        print(dataset)
        for model in MODELS:
            try:
                tstats = pd.read_csv(f"{OUT_PATH}/stats_{dataset}_{model}.csv")
                agg_mean.loc[
                    i * len(metric_ids):(i + 1) * len(metric_ids) - 1, model
                ] = tstats.loc[:, metric_ids].mean(axis=0).values
                agg_std.loc[
                    i * len(metric_ids):(i + 1) * len(metric_ids) - 1, model
                ] = tstats.loc[:, metric_ids].std(axis=0).values
            except FileNotFoundError:
                agg_mean.loc[
                    i * len(metric_ids):(i + 1) * len(metric_ids) - 1, model
                ] = float("NaN")
                agg_std.loc[
                    i * len(metric_ids):(i + 1) * len(metric_ids) - 1, model
                ] = float("NaN")

    final_table_name = f"{output_folder}/tree_stats.tex"
    with open(final_table_name, 'w') as f:
        # Preamble portion
        f.write("\\begin{table*}[!htbp]\n")
        f.write("\t\\caption{Tree stats}\n")
        f.write("\t\\label{tab_tree_stats}\n")
        f.write("\t\\centering\n")
        f.write("\t\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\t\\resizebox{\\textwidth}{!}{\n")
        f.write("\t\\begin{{tabular}}{{ll{0}}}\n".format(len(MODELS) * "r"))
        f.write("\t\t\\toprule\n")

        # Header portion
        header = "\t\tDataset & Metrics & "
        header += f"{' & '.join(MODELS.keys())}"
        header = '{}\\\\\n'.format(header)
        f.write(header)
        f.write('\t\t\\midrule\n')

        # Data portion
        n_alg = len(MODELS)

        avg_ranks = np.zeros((len(pmetrics), len(MODELS)))
        synth_ranks = np.zeros((len(pmetrics), len(MODELS)))
        real_ranks = np.zeros((len(pmetrics), len(MODELS)))
        n_synth = 0
        n_real = 0

        for i, (dataset_id, dataset) in enumerate(DATASETS.items()):
            for j in range(len(pmetrics)):
                if j == 0:
                    line = [f"\\multirow{{{len(metrics_names)}}}{{*}}{{{dataset}}}"]
                else:
                    line = [" "]

                line.append(pmetrics[j])

                row = agg_mean.iloc[i * len(pmetrics) + j, :]

                temp = np.argsort(row)
                ranks_ = np.zeros(avg_ranks.shape[1])
                ranks_[temp] = np.asarray(
                    [r if not np.isnan(val) else n_alg
                     for r, val in zip(np.arange(n_alg) + 1, row[temp])]
                )

                avg_ranks[j, :] += ranks_

                if dataset_id in SYNTH_DATA:
                    n_synth += 1
                    synth_ranks[j, :] += ranks_
                else:
                    n_real += 1
                    real_ranks[j, :] += ranks_

                for k in range(n_alg):
                    # For missing results
                    if np.isnan(agg_mean.iloc[i * len(pmetrics) + j, k]):
                        line.append("--")
                        continue

                    flag = k == np.nanargmin(agg_mean.iloc[i * len(pmetrics) + j, :].values)

                    if flag:
                        if INCLUDE_STD_IN_TABLES:
                            line.append(
                                "$\\mathbf{{{0:.2f} \\pm {1:.2f}}}$".format(
                                    agg_mean.iloc[i * len(pmetrics) + j, k],
                                    agg_std.iloc[i * len(pmetrics) + j, k]
                                )
                            )
                        else:
                            line.append(
                                "$\\mathbf{{{0:.2f}}}$".format(
                                    agg_mean.iloc[i * len(pmetrics) + j, k]
                                )
                            )
                    else:
                        if INCLUDE_STD_IN_TABLES:
                            line.append(
                                "${0:.2f} \\pm {1:.2f}$".format(
                                    agg_mean.iloc[i * len(pmetrics) + j, k],
                                    agg_std.iloc[i * len(pmetrics) + j, k]
                                )
                            )
                        else:
                            line.append(
                                "${0:.2f}$".format(
                                    agg_mean.iloc[i * len(pmetrics) + j, k]
                                )
                            )
                line = "\t\t{0}\\\\\n".format(" & ".join(line))
                f.write(line)
            f.write("\t\t\\midrule\n")

        avg_ranks /= len(DATASETS)
        if n_synth > 0 and n_real > 0:
            synth_ranks /= (n_synth / len(pmetrics))
            real_ranks /= (n_real / len(pmetrics))
            f.write(
                f"\t\t\\multirow{{{4 * (len(pmetrics))}}}{{*}}{{\\textbf{{Ranks}}}}"
            )
            for j in range(len(pmetrics)):
                f.write(
                    f"\t\t & {pmetrics[j]}{' & '.join(['' for _ in range(n_alg + 1)])}\\\\\n"
                )
                line1 = ["", "\\qquad\\textbf{Avg.}"]
                line2 = ["", "\\qquad\\textbf{Synth.}"]
                line3 = ["", "\\qquad\\textbf{Real}"]

                avg_min_pos = np.argmin(avg_ranks[j, :])
                synth_min_pos = np.argmin(synth_ranks[j, :])
                real_min_pos = np.argmin(real_ranks[j, :])

                for i, (a, s, r) in enumerate(
                    zip(avg_ranks[j, :], synth_ranks[j, :], real_ranks[j, :])
                ):
                    if i == avg_min_pos:
                        line1.append(f"$\\mathbf{{{a:.2f}}}$")
                    else:
                        line1.append(f"${a:.2f}$")

                    if i == synth_min_pos:
                        line2.append(f" $\\mathbf{{{s:.2f}}}$")
                    else:
                        line2.append(f" ${s:.2f}$")

                    if i == real_min_pos:
                        line3.append(f" $\\mathbf{{{r:.2f}}}$")
                    else:
                        line3.append(f"${r:.2f}$")
                f.write(f"\t\t{' & '.join(line1)}\\\\\n")
                f.write(f"\t\t{' & '.join(line2)}\\\\\n")
                f.write(f"\t\t{' & '.join(line3)}\\\\\n")

                if j < len(pmetrics) - 1:
                    f.write(f"\t\t\\cmidrule{{2-{len(MODELS) + 2}}}\n")
        else:
            for j in range(len(pmetrics)):
                if j == 0:
                    line = [f"\\multirow{{{len(pmetrics)}}}{{*}}\\textbf{{Ranks}}"]
                else:
                    line = [" "]
                line.append(pmetrics[j])

                avg_min_pos = np.argmin(avg_ranks[j, :])

                for i, a in enumerate(avg_ranks[j, :]):
                    if i == avg_min_pos:
                        cell = f"$\\mathbf{{{a:.2f}}}$"
                    else:
                        cell = f"${a:.2f}$"
                    line.append(cell)
                f.write(f"\t\t{' & '.join(line)}\n")

        f.write("\t\t\\bottomrule\n")
        f.write("\t\\end{tabular}}\n")
        f.write("\\end{table*}\n")


if __name__ == "__main__":
    output_folder = f"{OUT_PATH}/tables"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    main(output_folder)
