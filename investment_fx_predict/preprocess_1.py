import argparse
import numpy as np
import pandas as pd

from app_logging import get_app_logger
import app_s3
from app_datetime import week_ranges


def execute(fxpairs, years, input_base_path, output_base_path):
    L = get_app_logger()
    L.info(f"start: fxpairs={fxpairs}, years={years}")

    # Load data
    df_fxpairs = app_s3.read_dataframe(f"{input_base_path}/fxpairs.csv", index_col=0)
    df_result = pd.DataFrame(columns=["fxpair", "start_date", "end_date"])

    # Preprocess
    for id in df_fxpairs.index:
        fxpair = df_fxpairs.at[id, "fxpair"]
        year = df_fxpairs.at[id, "year"]

        if fxpair not in fxpairs or year not in years:
            continue

        result = preprocess(fxpair, year, input_base_path, output_base_path)
        if result["exception"] is not None:
            continue

        for week_range in result["week_ranges"]:
            result_id = len(df_result)

            df_result.at[result_id, "fxpair"] = fxpair
            df_result.at[result_id, "start_date"] = week_range[0]
            df_result.at[result_id, "end_date"] = week_range[1]

    # Save data
    app_s3.write_dataframe(df_result, f"{output_base_path}/fxpairs.csv")

    L.info("finish")


def preprocess(fxpair, year, input_base_path, output_base_path):
    L = get_app_logger()
    L.info(f"preprocess_1: fxpair={fxpair}, year={year}")

    result = {
        "fxpair": fxpair,
        "week_ranges": [],
        "exception": None
    }

    try:
        # Load data
        L.info("Load data")

        df = app_s3.read_dataframe(
            f"{input_base_path}/fx.{fxpair}.{year}.csv",
            dtype={"timestamp": str, "open": float, "high": float, "low": float, "close": float},
            parse_dates=["timestamp"]
        )

        # Preprocess
        L.info(f"{df.info()}")
        L.info(f"before start date: {df['timestamp'].min()}")
        L.info(f"before end date: {df['timestamp'].max()}")
        L.info(f"before records: {len(df)}")
        L.info(f"before nulls: {df.isnull().sum().sum()}")
        L.info(f"before duplicates date: {len(df)-len(df['timestamp'].unique())}")

        df = df.sort_values("timestamp") \
            .dropna() \
            .drop_duplicates(subset="timestamp")

        for week_range in week_ranges(year):
            df_week = df.query(f"'{week_range[0].strftime('%Y-%m-%d')} 00:00:00'<=timestamp<'{week_range[1].strftime('%Y-%m-%d')} 00:00:00'")

            df_week = df_week.assign(id=np.arange(len(df_week))) \
                .set_index("id")

            L.info(f"after start date: {df_week['timestamp'].min()}")
            L.info(f"after end date: {df_week['timestamp'].max()}")
            L.info(f"after records: {len(df_week)}")
            L.info(f"after nulls: {df_week.isnull().sum().sum()}")
            L.info(f"after duplicates date: {len(df_week)-len(df_week['timestamp'].unique())}")

            if len(df_week) == 0:
                continue

            # Save data
            L.info(f"Save data")

            app_s3.write_dataframe(df_week, f"{output_base_path}/fx.{fxpair}.{week_range[0].strftime('%Y%m%d')}_{week_range[1].strftime('%Y%m%d')}.csv")

            result["week_ranges"].append(week_range)
    except Exception as err:
        L.exception(f"fxpair={fxpair}, year={year}, {err}")
        result["exception"] = err

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    execute(
        fxpairs=["usdjpy"],
        years=[2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000],
        input_base_path="raw_data",
        output_base_path=f"preprocess_1"
    )
