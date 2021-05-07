import os
from utils import DATASETS
from utils import CAT_FEATURES
from utils import GENERATOR_BASED
from utils import IN_PATH
from utils import OUT_PATH


def dataset_stats(fname):
    with open(fname) as f:
        line = f.readline()
        n_features = len(line.split(",")) - 1
        n_instances = 0
        while line:
            n_instances += 1
            line = f.readline()
        n_instances -= 1
    return n_instances, n_features


if not os.path.exists(f"{OUT_PATH}/tables"):
    os.makedirs(f"{OUT_PATH}/tables")

table_name = f"{OUT_PATH}/tables/data_info.tex"

print("Generating table for the datasets characteristics")

with open(table_name, "w") as tbl_f:
    tbl_f.write("%Add to preamble: \\usepackage{booktabs, makecell}\n")
    tbl_f.write("\\begin{table}[!htb]\n")
    tbl_f.write("\t\\centering\n")
    tbl_f.write("\t\\caption{Characteristics of the evaluated datasets.}\n")
    tbl_f.write("\t\\label{tab_datasets}\n")
    tbl_f.write("\t\\resizebox{0.45\\textwidth}{!}{\n")
    tbl_f.write("\t\\begin{tabular}{lrrr}\n")
    tbl_f.write("\t\t\\toprule\n")
    header = "\t\tDataset & \\#Instances" + \
        " & \\thead{\\#Numeric\\\\features}" + \
        " & \\thead{\\#Categorical\\\\features} \\\\\n"
    tbl_f.write(header)
    tbl_f.write("\t\t\\midrule\n")
    for dataset, datasetp in DATASETS.items():
        if dataset in GENERATOR_BASED:
            continue

        print("Processing", datasetp)
        dataset_path = os.path.join(f"{IN_PATH}/{dataset}.csv")
        n_instances, n_features = dataset_stats(dataset_path)
        n_categorical = 0
        if dataset in CAT_FEATURES:
            n_categorical = len(CAT_FEATURES[dataset])
        n_numerical = n_features - n_categorical
        entry = f"\t\t{datasetp} & {n_instances} & {n_numerical} & {n_categorical}\\\\\n"
        tbl_f.write(entry)

    tbl_f.write("\t\t\\bottomrule\n")
    tbl_f.write("\t\\end{tabular}}\n")
    tbl_f.write("\\end{table}")
