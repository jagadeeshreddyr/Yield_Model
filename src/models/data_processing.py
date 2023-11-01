import datetime as dt
import glob
import os
import pandas as pd
from pathlib import Path


def preprocessing_data(
    info_path: str, raw_data_path: str = None, columns: str = None
) -> pd.DataFrame:
    """
    Preporocessing of data

    Parameters
    ----------
    info_path : str
        The info file contains all the data of the farmers.
    raw_data_path : str, optional
        This path contains farmers sensor data.
    columns : str, optional
        This contains the column names to consider
    Returns
    -------
        DataFrame : Returns trainning data for model.
    """

    if raw_data_path is None:
        return pd.DataFrame()

    client = pd.read_csv(info_path)
    raw_data_paths = glob.glob(raw_data_path)
    stack = []

    for raw_path in raw_data_paths:
        #
        fid_name = Path(raw_path).parent.stem

        # Reading raw sensor data report csv as data
        data = pd.read_csv(raw_path)

        # formatting date
        data["date"] = data["Date time (time zone: Asia/Kolkata)"].apply(
            lambda x: pd.to_datetime(x).date()
        )

        ## if parameters are selected
        if columns:
            for sensor_parameter in columns:
                data = data.loc[(data[sensor_parameter] != "-")]

            data[columns] = data[columns].astype(float)

            data = data.drop(columns=["Date time (time zone: Asia/Kolkata)"])

            data = data[["date", *columns]]

            temp_humd_df = (
                data.groupby(["date"])
                .mean()
                .reset_index()[["Temperature (°C)", "Humidity (%)"]]
            )
        else:
            ## if parameters are not selected default is Rainfall windspeed and age

            data = data.loc[
                (data["Rain fall (mm)"] != "-") & (data["Wind Speed (kmh)"] != "-")
            ]
            data[["Rain fall (mm)", "Wind Speed (kmh)"]] = data[
                ["Rain fall (mm)", "Wind Speed (kmh)"]
            ].astype("float")

        rainfall_df = (
            data.copy()[["date", "Rain fall (mm)"]].groupby("date").sum().reset_index()
        )
        sowingdate, farmer_name, crop, yield_cr = client.loc[
            client.fid == int(fid_name), ["sowingdt", "Name", "Crop", "Actual Yield"]
        ].values[0]
        ## calculating age of crop from sowing date
        rainfall_df["days"] = rainfall_df["date"].apply(
            lambda x: (x - dt.datetime.strptime(sowingdate, "%Y-%m-%d").date()).days
        )

        """
        max rainfall code 
        """
        index_max_rain = rainfall_df["Rain fall (mm)"].idxmax()

        max_rain_series = rainfall_df.iloc[index_max_rain]

        # data.loc[data["date"] == max_rain_series[0]]["Wind Speed (kmh)"].max()

        if columns:
            dict_df = {
                "fid": fid_name,
                "Name": farmer_name,
                "Crop": crop,
                "Rain fall (mm)": max_rain_series["Rain fall (mm)"],
                "Age": max_rain_series["days"],
                "Temp_min": temp_humd_df["Temperature (°C)"].min(),
                "Temp_max": temp_humd_df["Temperature (°C)"].max(),
                "Temp_mean": temp_humd_df["Temperature (°C)"].mean(),
                "Humd_min": temp_humd_df["Humidity (%)"].min(),
                "Humd_max": temp_humd_df["Humidity (%)"].max(),
                "Humd_mean": temp_humd_df["Humidity (%)"].mean(),
                "Wind Speed (kmh)": data.loc[data["date"] == max_rain_series["date"]][
                    "Wind Speed (kmh)"
                ].max(),
                "Actual_yield": yield_cr,
            }
        else:
            dict_df = {
                "fid": fid_name,
                "Name": farmer_name,
                "Crop": crop,
                "Rain fall (mm)": max_rain_series["Rain fall (mm)"],
                "Age": max_rain_series["days"],
                "Wind Speed (kmh)": data.loc[data["date"] == max_rain_series["date"]][
                    "Wind Speed (kmh)"
                ].max(),
                "Actual_yield": yield_cr,
            }

        # stack dictonarys as rows in list
        stack.append(dict_df)

    ## create final df
    df = pd.DataFrame(stack).round(2)

    ##trying to
    df.loc[df.fid == "68528", "Age"] = 130

    return df


if __name__ == "__main__":
    """
    To set the path to current directory
    """
    dirname = "/home/satyukt/Projects/5020/crop-yield/src/models"
    dirname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dirname)

    # info csv contains the data all the farmers
    info_path = "../../data/info/all_info.csv"

    # raw sensor data of all farmers
    raw_data_path = "../../data/raw/*/report.csv"

    # raw_data_path = "/home/satyukt/Projects/5020_1/data/1st batch of farm data july to oct 16/*/report.csv"

    # selected best coreleation sensor parameters
    columns = ["Rain fall (mm)", "Wind Speed (kmh)", "Temperature (°C)", "Humidity (%)"]

    """
    if columns are not selected default column is max rainfall, age of crop
    and windspeed
    """
    columns = None

    df = preprocessing_data(info_path, raw_data_path, columns)

    # after preprocessing training csv
    df.to_csv("../../data/processed/training.csv", index=False)
