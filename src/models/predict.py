import os

import pandas as pd
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")


def error(df: pd.DataFrame, x: str, y: str) -> tuple[float, float, float, float, float]:
    """
    this is to calculate the error

    Parameters
    ----------
    df : Dataframe
         This df contains dataframe contains predicted and actual yield

    x: str

        actual yield value column name

    y: str

        predicated yield value column name

    Returns
    -------
        mae = Mean absoulte error
        mse = Mean squared error
        var = Variance in model
        corr = correlation of model
        bias = Bias of the model
        r2 = Coefficient of determination
    """

    mae = mean_absolute_error(df[x], df[y]).round(2)
    mse = mean_squared_error(df[x], df[y]).round(2)
    var = df[y].var().round(2)
    corr = round(df[x].corr(df[y]), 2)
    bias = (df[x] - df[y]).sum() / (df[y].sum()) * 100
    r2 = r2_score(df[x], df[y]).round(3)

    return mae, mse, var, corr, bias, r2


if __name__ == "__main__":
    """
    To set the path to current directory
    """

    dirname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dirname)

    crop = "Chili"

    #   crop = 'Tomato'

    info_csv = pd.read_csv("../../data/processed/training.csv")

    df = info_csv.loc[info_csv["Crop"] == crop]

    ## Load trained scaler model
    with open(f"../../model/{crop}/slr.pkl", "rb") as file:
        model_scaler = pickle.load(file)

    ## Load trained model
    with open(f"../../model/{crop}/main.pkl", "rb") as file:
        model = pickle.load(file)

    ## input dataset for model
    dataset = df.iloc[:, 3:-1].values

    for ind, data in enumerate(df.index):
        fid = df.iloc[ind]["fid"]
        name = df.iloc[ind]["Name"]

        ## final model
        in1 = model_scaler.transform([dataset[ind]])

        df.at[data, "Predicted"] = model.predict(in1)

    df.to_csv(f"../../output/results/{crop}.csv", index=False)

    print(df)

    (
        mae,
        mse,
        var,
        corr,
        bias,
        r2,
    ) = error(df, "Actual_yield", "Predicted")

    print(
        f"crop = {crop}, Mean absolute error = {mae}, correleation = {corr}, r2 = {r2}"
    )
