import datetime as dt
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import warnings


from pathlib import Path


warnings.filterwarnings("ignore")
plt.style.use("ggplot")


def format_date(x: str, year=False) -> dt.datetime.strftime:
    """
    Formatting date if year is true it will return with year or
    only month and day

    x: str

        date in strings

    Return:

        formated date strptime format

    """

    if year == True:
        return dt.datetime.strptime(" ".join(x.split(" ")[:-2]), "%B %d, %Y").strftime(
            "%Y-%m-%d"
        )

    return dt.datetime.strptime(x.split(",")[0], "%B %d").strftime("%b %d")


def visualize_data_plot(sensor_path: str) -> None:
    """
    Preporocessing of sensor data (gdd, vpd, et) and plots the parameter with date.

    Parameters
    ----------
    sensor_path : str
        The info file contains all the sensor data (gdd, vpd, rt) of farmers.

    Returns
    -------
        None
    """

    sensor_files = glob.glob(sensor_path)

    for filename in sensor_files:
        path_name = Path(filename)
        name = path_name.parent.stem

        df = pd.read_csv(filename)
        df["Date time (time zone: Asia/Kolkata)"] = df[
            "Date time (time zone: Asia/Kolkata)"
        ].apply(lambda x: format_date(x))
        df = df.rename(columns={"Date time (time zone: Asia/Kolkata)": "Datetime"})

        df["Datetime"] = df["Datetime"].apply(
            lambda x: dt.datetime.strptime(x, "%b %d").strftime("%m-%d")
        )
        parameter = df.columns[1]

        if parameter in ["VPD", "Temperature (°C)", "Humidity (%)"]:
            df = df.loc[df[parameter] != "-"]

            df = df.groupby("Datetime").mean().reset_index()

        fig, ax = plt.subplots()
        df.plot(df.columns[0], df.columns[1], ax=ax)
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.title(f"{parameter} - {name}")

        if parameter == "Evapotranspiration (mm/day)":
            parameter = "Evapotranspiration"
        nested_directory_path = Path(
            f"../../output/visual/sensor/{name}/{parameter}.png"
        )
        nested_directory_path.parent.mkdir(exist_ok=True)
        plt.savefig(nested_directory_path)


def plot_report_csv(report_path: str) -> None:
    """
    Preporocessing of report csv and plots the parameter with days

    Parameters
    ----------
    report_path : str
        The info file contains all the report csv data of farmers.

    Returns
    -------
        None
    """

    client_info = pd.read_csv("../../data/info/all_info.csv")

    sensor_files = glob.glob(report_path)

    for filename in sensor_files:
        path_name = Path(filename)
        name = path_name.parent.stem

        df = pd.read_csv(filename)

        # This is with year
        df["Date"] = df["Date time (time zone: Asia/Kolkata)"].apply(
            lambda x: format_date(x, year=True)
        )

        # this formatting only with month and date
        df["Date time (time zone: Asia/Kolkata)"] = df[
            "Date time (time zone: Asia/Kolkata)"
        ].apply(lambda x: format_date(x))
        df = df.rename(columns={"Date time (time zone: Asia/Kolkata)": "Datetime"})

        """
        This below parameters are plotted with month-day vs parameters
        """

        l1_parameters = [
            "Temperature (°C)",
            "Humidity (%)",
            "Pressure (hPa)",
            "Soil temperature (°C)",
        ]

        for parameter in l1_parameters:
            df1 = df[["Datetime", parameter]]

            df1 = df1.loc[df1[parameter] != "-"]
            df1[parameter] = df1[parameter].astype(float)

            df1 = df1.groupby("Datetime").mean().reset_index()
            df1["Datetime"] = df1["Datetime"].apply(
                lambda x: dt.datetime.strptime(x, "%b %d").strftime("%m-%d")
            )

            fig, ax = plt.subplots()
            df1.plot("Datetime", parameter, ax=ax)
            plt.xlabel(df1.columns[0])
            plt.ylabel(df1.columns[1])
            plt.title(f"{parameter} - {name}")

            nested_directory_path = Path(
                f"../../output/visual/sensor/{name}/{parameter}.png"
            )
            nested_directory_path.parent.mkdir(exist_ok=True)
            plt.savefig(nested_directory_path)

        """
        These below parameters will scatter plotted days vs parameter
        """

        l2_parameters = ["Wind Speed (kmh)", "Rain fall (mm)", "Leaf wetness (%)"]

        for parameter in l2_parameters:
            df1 = df[["Date", parameter]]
            df1["Date"] = pd.to_datetime(df1["Date"])
            sw = client_info.loc[client_info.fid == int(name), "sowingdt"].values[0]
            df1["days"] = df1["Date"].apply(
                lambda x: (x - dt.datetime.strptime(sw, "%Y-%m-%d")).days
            )

            df1 = df1.loc[df1[parameter] != "-"]
            df1[parameter] = df1[parameter].astype(float)

            df1 = df1.loc[df1.days > 0]

            df1 = df1.drop(columns=["Date"])

            df1 = df1.groupby("days").sum().reset_index()

            fig, ax = plt.subplots(figsize=(10, 6))

            df1.plot.scatter("days", parameter, ax=ax)
            plt.xlabel("Days")
            plt.ylabel(df1.columns[1])
            plt.title(f"{parameter} - {name}")
            nested_directory_path = Path(
                f"../../output/visual/sensor/{name}/{parameter}.png"
            )
            nested_directory_path.parent.mkdir(exist_ok=True)
            plt.savefig(nested_directory_path)


def correlation_heatmap(report_path):
    sensor_files = glob.glob(report_path)
    for filename in sensor_files:
        path_name = Path(filename)
        name = path_name.parent.stem
        df = pd.read_csv(filename)
        df = df.rename(
            columns={
                "Date time (time zone: Asia/Kolkata)": "Datetime",
                "Temperature (°C)": "temp(°C)",
                "Humidity (%)": "Humidity(%)",
                "Pressure (hPa)": "Pressure(hPa)",
                "Wind direction": "Winddirection",
                "Wind Speed (kmh)": "Wind Speed(kmh)",
                "Rain fall (mm)": "Rainfall(mm)",
                "Leaf wetness (%)": "Lf-wetness(%)",
                "Soil temperature (°C)": "Soil-temp(°C)",
                "Soil moisture primary root zone (centibar)": "SM primary",
                "Soil moisture secondary root zone (centibar)": "SM secoundary",
                "Solar intensity (%)": "Solar-intensity(%)",
            }
        )
        cmap = sns.diverging_palette(500, 10, as_cmap=True)
        df = df.drop(columns=["Datetime", "Winddirection"])
        corr = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, linewidths=0.5, cmap="Blues", center=0)
        plt.xticks(rotation=45)
        nested_directory_path = Path(f"../../output/visual/sensor/{name}/heatmap.png")
        os.makedirs(os.path.dirname(nested_directory_path), exist_ok=True)
        plt.savefig(nested_directory_path)


if __name__ == "__main__":
    """
    To set the path to current directory
    """

    dirname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dirname)

    gdd_path = "../../data/raw/*/gdd.csv"

    vpd_path = "../../data/raw/*/vpd.csv"

    et_path = "../../data/raw/*/evapotranspiration.csv"

    report_path = "../../data/raw/*/report.csv"

    """
    pass the path which plots you want see

    """

    correlation_heatmap(report_path)

    plot_report_csv(report_path)

    for i in [et_path, gdd_path, vpd_path]:
        visualize_data_plot(i)
