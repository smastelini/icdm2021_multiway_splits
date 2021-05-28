import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from utils import DATASETS, MODELS, OUT_PATH


metric_prettyfier = {
    "RMSE": "RMSE",
    "memory": "Memory (MB)",
    "time": "Time (s)"
}

model_prettyfier = {
    "HTR + QO$_{0.1} + M$": "HTR + QO$_{0.1}^M$",
    "HTR + QO$_{0.25} + M$": "HTR + QO$_{0.25}^M$",
    "HTR + QO$_{0.5} + M$": "HTR + QO$_{0.5}^M$",
    "HTR + QO$_{1} + M$": "HTR + QO$_{1}^M$",
}

raw_results = {
    metric: pd.DataFrame(
        np.zeros((len(DATASETS), len(MODELS))),
        columns=list(MODELS)
    ) for metric in metric_prettyfier
}

for i, dataset in enumerate(DATASETS):
    for j, model in enumerate(MODELS):
        for metric in metric_prettyfier:
            log = pd.read_csv(f"{OUT_PATH}/final/mean_{dataset}_{model}.csv")
            raw_results[metric].iloc[i, j] = log.tail(1)[metric].values[0]

proc_results = pd.DataFrame(
    np.zeros((len(MODELS), len(metric_prettyfier))),
    columns=list(metric_prettyfier.values())
)

proc_results["variant"] = [model_prettyfier.get(name, name) for name in MODELS.keys()]

for metric, metricp in metric_prettyfier.items():
    for i in range(len(raw_results[metric])):
        row = raw_results[metric].iloc[i, :].values
        # Put all values between 0 and 1
        raw_results[metric].iloc[i, :] = (row - min(row)) / (max(row) - min(row))

    proc_results[metricp] = raw_results[metric].mean(axis=0).values


scale = StandardScaler()
X = scale.fit_transform(proc_results.iloc[:, :-1])

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

df_pca = pd.DataFrame(X_pca)
df_pca["QO variant"] = [modeln.replace("HTR + ", "") for modeln in proc_results.variant]
df_pca.columns = ["PC1", "PC2", "PC3", "QO variant"]

cmap = mpl.cm.RdYlBu(np.linspace(0, 1, len(df_pca)))
cmap[:, :3] *= 0.75
cmap = mpl.colors.LinearSegmentedColormap.from_list("darker_biplot", cmap)
plt.register_cmap(cmap=cmap)

markers = ["o", "X", "^", "<", ">", "8", "s", "p", "P", "*"][:len(MODELS)]

fig, ax = plt.subplots(figsize=(8, 5), dpi=600)

scalex = 1.0 / (df_pca["PC1"].max() - df_pca["PC1"].min())
scaley = 1.0 / (df_pca["PC2"].max() - df_pca["PC2"].min())

xvector = scalex * df_pca["PC1"]
yvector = scaley * df_pca["PC2"]

g = sns.scatterplot(
    x=xvector, y=yvector, hue=df_pca["QO variant"], s=100,
    ax=ax, palette="darker_biplot", markers=markers, style=df_pca["QO variant"]
)

coeff = np.transpose(pca.components_[0:2, :])
n = coeff.shape[0]

for i in range(n):
    # arrows project features (i.e. columns from csv) as vectors onto PC axes
    x_coord = coeff[i, 0]
    y_coord = coeff[i, 1]
    ax.arrow(0, 0, x_coord, y_coord,
             color="r", width=0.005, head_width=0.02)
    ax.text(
        x_coord + 0.05,
        y_coord if i == 0 else (y_coord + 0.1 if i == 1 else y_coord - 0.1),
        list(proc_results)[i],
        color="r"
    )

# ax.set_xlim(
#     [round(min(1.22 * min(coeff[:, 0]), min(xvector))) - 0.3,
#      round(max(1.22 * max(coeff[:, 0]), max(xvector))) + 0.3]
# )
# ax.set_ylim(
#     [round(min(1.22 * min(coeff[:, 1]), min(yvector))) - 0.1,
#      round(max(1.22 * max(coeff[:, 1]), max(yvector))) + 0.1]
# )

ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))

ax.set_xlabel("PC1 ({0:.2f}%)".format(100 * pca.explained_variance_ratio_[0]))
ax.set_ylabel("PC2 ({0:.2f}%)".format(100 * pca.explained_variance_ratio_[1]))

ax.axvline(
    0.0, linestyle="--", linewidth="0.5", color="gray", zorder=-1, alpha=0.5
)
ax.axhline(
    0.0, linestyle="--", linewidth="0.5", color="gray", zorder=-1, alpha=0.5
)

# Twin axes
# ax_ = ax.twinx()
# ax_.set_yticks(np.linspace(round(coeff.min()), round(coeff.max()), 5))
# ax_.tick_params(axis="y", labelcolor="red")
# ax_ = ax.twiny()
# ax_.set_xticks(np.linspace(round(coeff.min()), round(coeff.max()), 5))
# ax_.tick_params(axis="x", labelcolor="red")

g.get_legend().set_title("QO variants")
plt.setp(g.get_legend().get_texts(), fontsize="12.5")
plt.setp(g.get_legend().get_title(), fontsize="12.5")

plt.savefig(f"{OUT_PATH}/charts/QO_biplot.png", bbox_inches="tight")
