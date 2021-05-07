from river import linear_model
from river import neighbors
from river import tree


IN_PATH = "../datasets"
OUT_PATH = "../output"

MAIN_SEED = 42
INCLUDE_STD_IN_TABLES = True
INCLUDE_BASELINES_IN_TABLES = False

DATASETS = {
    "abalone": "Abalone",
    "ailerons": "Ailerons",
    # "airlines07_08": "Airlines 07-08",
    "bike": "Bike",
    "cal_housing": "CalHousing",
    "elevators": "Elevators",
    "house_8L": "House8L",
    "house_16H": "House16H",
    "metro_interstate_traffic": "Metro",
    # "msd_year": "MSD Year",
    "pol": "Pol",
    "wind": "Wind",
    "winequality": "Wine",
    # # synthetic
    "friedman": "Friedman",
    "mv": "MV",
    "puma8NH": "Puma8NH",
    "puma32H": "Puma32H",
}

SYNTH_DATA = {
    "friedman",
    "mv",
    "puma8NH",
    "puma32H",
}

GENERATOR_BASED = {
    "friedman",
    "mv",
}

CAT_FEATURES = {
    "abalone": [0],
    "airlines07_08": [1, 2, 3, 6, 9, 10, 12],
    "metro_interstate_traffic": [0, 5, 6],
    "wind": [1, 2]
}

N_REPS = 5


MODELS = {
    "HTR + E-BST": tree.HoeffdingTreeRegressor(
        leaf_prediction="mean"
    ),
    "HTR + TE-BST": tree.HoeffdingTreeRegressor(
        leaf_prediction="mean",
        splitter=tree.splitter.TEBSTSplitter(digits=2),
    ),
    "HTR + QO$_{0.1}$": tree.HoeffdingTreeRegressor(
        leaf_prediction="mean",
        splitter=tree.splitter.QOSplitter(radius=0.1)
    ),
    "HTR + QO$_{0.25}$": tree.HoeffdingTreeRegressor(
        leaf_prediction="mean",
        splitter=tree.splitter.QOSplitter(radius=0.25)
    ),
    "HTR + QO$_{0.5}$": tree.HoeffdingTreeRegressor(
        leaf_prediction="mean",
        splitter=tree.splitter.QOSplitter(radius=0.5)
    ),
    "HTR + QO$_1$": tree.HoeffdingTreeRegressor(
        leaf_prediction="mean",
        splitter=tree.splitter.QOSplitter(radius=1)
    ),
    "HTR + QO$_{0.1} + M$": tree.HoeffdingTreeRegressor(
        leaf_prediction="mean",
        splitter=tree.splitter.QOSplitter(radius=0.1, allow_multiway_splits=True)
    ),
    "HTR + QO$_{0.25} + M$": tree.HoeffdingTreeRegressor(
        leaf_prediction="mean",
        splitter=tree.splitter.QOSplitter(radius=0.25, allow_multiway_splits=True)
    ),
    "HTR + QO$_{0.5} + M$": tree.HoeffdingTreeRegressor(
        leaf_prediction="mean",
        splitter=tree.splitter.QOSplitter(radius=0.5, allow_multiway_splits=True)
    ),
    "HTR + QO$_{1} + M$": tree.HoeffdingTreeRegressor(
        leaf_prediction="mean",
        splitter=tree.splitter.QOSplitter(radius=1, allow_multiway_splits=True)
    ),
}


BASELINES = {
    "LR": linear_model.LinearRegression(),
    "PAR": linear_model.PARegressor(),
    #"k-NN": neighbors.KNNRegressor()
}

if INCLUDE_BASELINES_IN_TABLES:
    MODELS.update(BASELINES)
