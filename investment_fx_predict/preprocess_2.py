import argparse
import pandas as pd
from sklearn.preprocessing import scale

from app_logging import get_app_logger
import app_s3


def execute(fxpairs, years, input_base_path, output_base_path):
    L = get_app_logger()
    L.info(f"start: fxpairs={fxpairs}, years={years}")

    # Load data
    df_fxpairs = app_s3.read_dataframe(
        f"{input_base_path}/fxpairs.csv",
        dtype={"id": int, "fxpair": str, "start_date": str, "end_date": str},
        parse_dates=["start_date", "end_date"],
        index_col=0
    )
    df_result = pd.DataFrame(columns=df_fxpairs.columns)

    # Preprocess
    for id in df_fxpairs.index:
        if df_fxpairs.at[id, "fxpair"] not in fxpairs or df_fxpairs.at[id, "start_date"].year not in years:
            continue

        result = preprocess(df_fxpairs.at[id, "fxpair"], df_fxpairs.at[id, "start_date"], df_fxpairs.at[id, "end_date"], input_base_path, output_base_path)
        if result["exception"] is not None:
            continue

        df_result.loc[id] = df_fxpairs.loc[id]

    app_s3.write_dataframe(df_result, f"{output_base_path}/fxpairs.csv")

    L.info("finish")


def preprocess(fxpair, start_date, end_date, input_base_path, output_base_path):
    L = get_app_logger()
    L.info(f"preprocess_2: fxpair={fxpair}, start_date={start_date}, end_date={end_date}")

    result = {
        "fxpair": fxpair,
        "start_date": start_date,
        "end_date": end_date,
        "exception": None
    }

    try:
        # Load data
        L.info(f"Load data")

        df = app_s3.read_dataframe(
            f"{input_base_path}/fx.{fxpair}.{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
            dtype={"id": int, "timestamp": str, "open": float, "high": float, "low": float, "close": float},
            parse_dates=["timestamp"],
            index_col=0
        )

        # EMA (Exponential Moving Average)
        L.info("Calc EMA")

        ema_len_array = [5, 9, 15, 26, 40, 80]
        for ema_len in ema_len_array:
            df[f"ema_{ema_len}"] = df["close"].ewm(span=ema_len).mean()
            df[f"ema_{ema_len}_diff"] = df[f"ema_{ema_len}"].diff()
            df[f"ema_{ema_len}_std"] = scale(df[f"ema_{ema_len}_diff"])

        # Save
        L.info("Save data")

        app_s3.write_dataframe(df, f"{output_base_path}/fx.{fxpair}.{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")

        L.info(df.info())
    except Exception as err:
        L.exception(f"fxpair={fxpair}, {err}")
        result["exception"] = err

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    execute(
        fxpairs=["usdjpy"],
        years=[2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000],
        input_base_path=f"preprocess_1",
        output_base_path=f"preprocess_2",
    )
