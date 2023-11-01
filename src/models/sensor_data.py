import glob

import numpy as np
import os
from pathlib import Path
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


import warnings

warnings.filterwarnings("ignore")


def nuts_cond(nutrient_val, nutrient, crop=None):
    """
    to find optimal nutrients if its not within the range

     Parameters
    ----------
    nutrient_val : int
        selected  Nutrient value
    nutrient : str
        To point which nutrient it is.
    crop : str
        crop for selecting condtions
    Returns
    -------
        0 or optimal value
    """

    if crop == "Chili":
        optimal_nutrients = {
            "N": [280, 560],
            "P": [22.4, 56],
            "K": [110, 280],
            "pH": [5.6, 6.8],
        }

    else:
        optimal_nutrients = {
            "N": [280, 560],
            "P": [22.4, 56],
            "K": [110, 280],
            "pH": [6.5, 7.5],
        }

    if (
        nutrient_val > optimal_nutrients[nutrient][0]
        and nutrient_val < optimal_nutrients[nutrient][1]
    ):
        return 0

    else:
        mean_value = np.array(
            [optimal_nutrients[nutrient][0], optimal_nutrients[nutrient][1]]
        ).mean()

        return ((nutrient_val - mean_value) / mean_value) * 0.25


def phases_condtion(df, crop=None):
    """
    creating condtions based phases and weights

    Parameters
    ----------
    df : DataFrame
        df with parameter and days
    crop : str
        crop for selecting condtions
    Returns
    -------
        Dataframe with phase weights
    """

    if crop == "Chili":
        conditions_ndvi = [
            df.days < 10,
            ((df.days >= 10) & (df.days < 30)),
            ((df.days >= 30) & (df.days < 90)),
            ((df.days >= 90)),
        ]

    else:
        conditions_ndvi = [
            df.days < 10,
            ((df.days >= 10) & (df.days < 30)),
            ((df.days >= 30) & (df.days < 90)),
            ((df.days >= 90)),
        ]

    choices = ["Phase-1", "Phase-2", "Phase-3", "Phase-4"]
    weights = [0.15, 0.15, 0.35, 0.35]
    df["Phase"] = np.select(conditions_ndvi, choices)
    df["Phase_weight_"] = np.select(conditions_ndvi, weights)

    return df


def rain_wind_condtion(df, column, col2=None):
    """
    This function is optimal condtions for rainfall and windspeed


    Parameters
    ----------
    df : DataFrame
        df with parameter and days
    column : str
        for selecting condtions based on parameter
    col2:
        used while creating condtions for optimal rainfall and windspeed

    Returns
    -------
        Dataframe with weights based on condtions
    """

    if (column == "Rain fall (mm)") or (col2 == "Rain fall (mm)"):
        condition = [
            (df[column] < 25),
            (df[column] >= 25) & (df[column] < 50),
            (df[column] >= 50) & (df[column] < 100),
            (df[column] >= 100),
        ]

        choices = [1, 0.9, 0.75, 0.5]

    else:
        condition = [
            (df[column] < 9.6),
            ((df[column] >= 9.6) & (df[column] < 14.5)),
            (df[column] >= 14.5),
        ]

        choices = [1, 0.8, 0.5]

    return np.select(condition, choices)


def scale_sensor_values(row):
    """
    we apply this function row wise and used for Temperature and Humidty

    Parameters
    ----------
    row : dataframe row
        this will select row wise data
    Returns
    -------
        optimal value of each row value
    """

    try:
        parameter = row[col_name]

    except:
        parameter = row[row.index[1]]

    min_val = row[min]
    max_val = row[max]

    if (parameter >= min_val) & (parameter <= max_val):
        return 1

    if parameter < min_val:
        return parameter / min_val

    if parameter > max_val:
        return max_val / parameter


def conditions_temp_humd(df, crop, column):
    """
    optimal condtions for Temperature and Humidty

    Parameters
    ----------
    df : Dataframe
        df of a farm

    crop: str
        based on crop days and condtions will change

    column:str
        either Humidity (%) or Temperature (°C)

    Returns
    -------
        Dataframe with Phasewise mean loss
    """

    if crop == "Chili":
        ct = [
            (df["days"] >= 0) & (df["days"] < 30),
            (df["days"] >= 30) & (df["days"] < 75),
            (df["days"] >= 75) & (df["days"] < 165),
            (df["days"] >= 165),
        ]

    else:
        ct = [
            (df["days"] >= 0) & (df["days"] < 10),
            (df["days"] >= 10) & (df["days"] < 30),
            (df["days"] >= 30) & (df["days"] < 90),
            (df["days"] >= 90),
        ]

    global min, max, col_name

    col_name = column

    if col_name == "Temperature (°C)":
        min = "temp_min"
        max = "temp_max"
        min_parameter = [15, 18, 15, 20]
        max_parameter = [30, 32, 32, 30]

    else:
        min = "humd_min"
        max = "humd_max"

        min_parameter = [40, 40, 40, 40]
        max_parameter = [75, 90, 90, 85]

    df[min] = np.select(ct, min_parameter)
    df[max] = np.select(ct, max_parameter)

    df["loss"] = df.apply(scale_sensor_values, axis=1)

    df = phases_condtion(df, crop)

    df = df.groupby("Phase").mean()[["loss"]]

    return df


def run(raw_sensor_data, crop, condtions=None):
    """
    Data preprocessing, finding optimal condtions for Rainfall, Temperature, Windspeed and Humidty
    and based on these creating losses and training Convolution neural network with these model predicting losses
    and predicting yield.

    Parameters
    ----------
    raw_sensor_data : list
        raw sensor data files

    crop: str
        Tomato and Chili for condtions

    Returns
    -------
        None
    """

    client_info = pd.read_csv("../../data/info/all_info.csv")

    crop_fids = client_info.loc[client_info.Crop == crop, "fid"].values.tolist()

    fids_list = []

    rainfall_stack = []

    temp_stack = []

    humd_stack = []

    wind_stack = []

    for sensor_data in raw_sensor_data:
        fid = Path(sensor_data).parent.stem

        if int(fid) in crop_fids:
            fids_list.append(fid)
            data = pd.read_csv(sensor_data)
            sow = client_info.loc[client_info.fid == int(fid), "sowingdt"].values[0]
            data["date"] = pd.to_datetime(data["Date time (time zone: Asia/Kolkata)"])
            data = data.loc[data["date"] >= pd.to_datetime(sow)]
            data["days"] = data.date.apply(lambda x: (x - pd.to_datetime(sow)).days)
            data = data.loc[
                (data["Rain fall (mm)"] != "-")
                & (data["Humidity (%)"] != "-")
                & (data["Temperature (°C)"] != "-")
                & (data["Wind Speed (kmh)"] != "-")
            ]
            data = data[
                [
                    "days",
                    "Rain fall (mm)",
                    "Humidity (%)",
                    "Wind Speed (kmh)",
                    "Temperature (°C)",
                ]
            ]

            data[
                [
                    "Rain fall (mm)",
                    "Humidity (%)",
                    "Wind Speed (kmh)",
                    "Temperature (°C)",
                ]
            ] = data[
                [
                    "Rain fall (mm)",
                    "Humidity (%)",
                    "Wind Speed (kmh)",
                    "Temperature (°C)",
                ]
            ].astype(
                float
            )

            """rainfall sensor data"""

            rain_df = data[["days", "Rain fall (mm)"]]

            rain_df["rain_loss"] = rain_wind_condtion(rain_df, "Rain fall (mm)")

            rain_df = phases_condtion(rain_df)

            rain_df = rain_df.groupby("Phase").min()

            rainfall_stack.append(rain_df["rain_loss"].values.tolist())

            """temperaure sensor data"""

            temp_df = data[["days", "Temperature (°C)"]]

            temp_stack.append(
                conditions_temp_humd(temp_df, crop, "Temperature (°C)")[
                    "loss"
                ].values.tolist()
            )

            """humidty sensor data"""

            humd_df = data[["days", "Humidity (%)"]]

            humd_stack.append(
                conditions_temp_humd(humd_df, crop, "Humidity (%)")[
                    "loss"
                ].values.tolist()
            )

            """windspeed sensor data"""

            wind_df = data[["days", "Wind Speed (kmh)"]]

            wind_df["wind_loss"] = rain_wind_condtion(wind_df, "Wind Speed (kmh)")

            wind_df = phases_condtion(wind_df)

            wind_df = wind_df.groupby("Phase").min()

            wind_stack.append(wind_df["wind_loss"].values.tolist())

    # rain_sensor = rainfall_stack
    # temp_sensor = temp_stack
    # humd_sensor = humd_stack
    # wind_sensor = wind_stack

    """optimal rainfall and windspeed"""

    optimal_rainfall = pd.read_csv(
        "../../data/optimal_sensor/rainfall_training_data.csv"
    )
    optimal_windspeed = pd.read_csv(
        "../../data/optimal_sensor/windspeed_training_data.csv"
    )

    optimal_rainfall_df = pd.DataFrame()
    optimal_windspeed_df = pd.DataFrame()

    for i in optimal_windspeed.columns[1:].values:
        optimal_rainfall_df[i] = rain_wind_condtion(
            optimal_rainfall, i, "Rain fall (mm)"
        )
        optimal_windspeed_df[i] = rain_wind_condtion(optimal_windspeed, i)

    optimal_rainfall_df["days"] = optimal_rainfall.Day.values
    optimal_rainfall_df = phases_condtion(optimal_rainfall_df)
    optimal_rainfall_df = optimal_rainfall_df.groupby(["Phase"]).min().T[:-2]

    """final rainfall dataframe of both sensor and optimal"""

    final_rain_df = pd.concat(
        [
            pd.DataFrame(
                rainfall_stack, columns=["Phase-1", "Phase-2", "Phase-3", "Phase-4"]
            ),
            optimal_rainfall_df,
        ]
    ).fillna(1)

    optimal_windspeed_df["days"] = optimal_windspeed.Day.values
    optimal_windspeed_df = phases_condtion(optimal_windspeed_df)
    optimal_windspeed_df = optimal_windspeed_df.groupby(["Phase"]).min().T[:-2]

    """final windspeed dataframe of both sensor and optimal"""
    final_wind_df = pd.concat(
        [
            pd.DataFrame(
                wind_stack, columns=["Phase-1", "Phase-2", "Phase-3", "Phase-4"]
            ),
            optimal_windspeed_df,
        ]
    ).fillna(1)

    """optimal temperature"""
    optimal_temperature = pd.read_csv(
        "../../data/optimal_sensor//tavg_training_data.csv"
    )
    optimal_temperature = optimal_temperature.rename(columns={"Day": "days"})
    for i in optimal_temperature.columns[1:].values:
        temp_stack.append(
            conditions_temp_humd(
                optimal_temperature[["days", i]], crop, "Temperature (°C)"
            )["loss"].values
        )

    """final temperature dataframe of both sensor and optimal"""
    final_temp_df = pd.DataFrame(
        temp_stack, columns=["Phase-1", "Phase-2", "Phase-3", "Phase-4"]
    ).fillna(1)

    """optimal Humidty"""
    optimal_humidty = pd.read_csv("../../data/optimal_sensor//tavg_training_data.csv")
    optimal_humidty = optimal_humidty.rename(columns={"Day": "days"})
    for i in optimal_humidty.columns[1:].values:
        humd_stack.append(
            conditions_temp_humd(optimal_humidty[["days", i]], crop, "Humidity (%)")[
                "loss"
            ].values
        )

    """final humidty dataframe of both sensor and optimal"""
    final_humd_df = pd.DataFrame(
        humd_stack, columns=["Phase-1", "Phase-2", "Phase-3", "Phase-4"]
    ).fillna(1)

    """Lets club all data for final model"""

    hum_sim = final_humd_df.values.tolist()
    temp_sim = final_temp_df.values.tolist()
    rain_sim = final_rain_df.values.tolist()
    wind_sim = final_wind_df.values.tolist()

    sensor_data_combo = []

    for rain in rain_sim[:60]:
        for temp in temp_sim[:60]:
            for hum in hum_sim[:60]:
                for win in wind_sim[:60]:
                    sensor_data_combo.append([rain, temp, hum, win])

    """creating input data and losses with factors"""
    full_sim = np.array(sensor_data_combo)
    overall_loss = (
        ((full_sim * np.array([0.15, 0.15, 0.35, 0.35])).sum(axis=2))
        * (np.array([0.40, 0.2, 0.2, 0.2]))
    ).sum(axis=1)
    train_x = full_sim.reshape((len(full_sim), 4, 4, 1))
    train_y = overall_loss

    """Convolution Neural Network for final model"""
    model = Sequential()
    model.add(
        Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(4, 4, 1))
    )  # Input layer
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mean_absolute_error", optimizer="adam")

    model.fit(train_x, train_y, epochs=10, batch_size=100000)

    """creating data for prediction"""
    test_x = np.array(sensor_data_combo[: len(fids_list)])
    # test_y = (((test_x*np.array([0.15, 0.15, 0.35, 0.35])).sum(axis=2))*(np.array([0.40, 0.2, 0.2,0.2]))).sum(axis=1)
    test_x = test_x.reshape((len(test_x), 4, 4, 1))

    """Prediction"""
    pred = model.predict(test_x).flatten()
    final_loss = pd.DataFrame(list(zip(fids_list, pred)), columns=["fid", "Losses"])
    final_loss["fid"] = final_loss["fid"].astype(int)

    # merging losses and Nutrients data with there respective farm id
    npk_data = pd.read_csv("../../data/info/fertilizer_impact.csv")

    final_loss = final_loss.merge(
        npk_data[["fid", "N", "P", "K", "pH"]], on="fid"
    ).merge(client_info[["fid", "Name", "Actual Yield"]], on="fid")

    """
    optimal losses based on npk data and optimal factors and prediction of Yield
    
    """

    final_loss["opti_nutrient"] = 0

    for index in range(len(final_loss)):
        for nutrients in ["N", "P", "K", "pH"]:
            final_loss.at[index, "opti_nutrient"] += nuts_cond(
                final_loss.iloc[index][nutrients], nutrients, crop
            )

        final_loss.at[index, "opti_nutrient"] = (
            1 + final_loss.at[index, "opti_nutrient"]
        )

        final_loss.at[index, "Predicted_Yield"] = (
            final_loss.at[index, "opti_nutrient"]
            * final_loss.at[index, "Losses"]
            * final_loss.at[index, "Actual Yield"]
        )

    print(final_loss[["fid", "Name", "Actual Yield", "Predicted_Yield"]])


if __name__ == "__main__":
    """
    To set the path to current directory
    """

    dirname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dirname)

    """List all sensor data"""
    sensor_data_files = glob.glob("../../data/raw/*/report.csv")

    seed_type = {
        "US440": [1.25, [5, 17]],
        "US800 F1 hybrid": [1.35, [6, 22]],
        "PHC 448": [1.05, [4.5, 15]],
        "HM CLAS prishi": [1.4, [6, 25]],
        "Syngenta": [1.5, [9, 25]],
        "General": [1, [4, 15]],
    }

    # select any one crop
    crop = "Tomato"
    crop = "Chili"

    run(sensor_data_files, crop)
