import argparse
import pandas as pd

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
    L.info(f"preprocess_4: fxpair={fxpair}, start_date={start_date}, end_date={end_date}")

    result = {
        "fxpair": fxpair,
        "start_date": start_date,
        "end_date": end_date,
        "exception": None
    }

    period = 60
    losscut_rate = 0.998
    take_profit_rate = 1.005

    try:
        # Load data
        L.info(f"Load data")

        df = app_s3.read_dataframe(
            f"{input_base_path}/fx.{fxpair}.{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
            dtype={"id": int, "timestamp": str, "open": float, "high": float, "low": float, "close": float},
            parse_dates=["timestamp"],
            index_col=0
        )

        # Max/Min price/rate
        L.info("Calc max/min profit")

        df["bid"] = df["close"] - 0.005
        df["ask"] = df["close"]

        df["max_price"] = df["bid"].rolling(period).max().shift(-period+1)
        df["max_profit_rate"] = (df["max_price"] - df["ask"]) / df["max_price"]
        df["min_price"] = df["bid"].rolling(period).min().shift(-period+1)
        df["min_profit_rate"] = (df["min_price"] - df["ask"]) / df["min_price"]

        # Profit rate
        L.info(f"Calc profit")

        for buy_id in df.index:
            buy_price = df.at[buy_id, "ask"]
            losscut_price = buy_price * losscut_rate
            take_profit_price = buy_price * take_profit_rate

            sell_id = None
            sell_price = None
            turn = 0

            for id in df.loc[buy_id+1:].index:
                # Sell: take profit
                if df.at[id, "bid"] > take_profit_price:
                    sell_id = id
                    sell_price = df.at[id, "bid"]
                    break

                # Sell: losscut
                if df.at[id, "low"] < losscut_price:
                    sell_id = id
                    sell_price = df.at[id, "low"]
                    break

                # Sell: time up
                if turn == period:
                    sell_id = id
                    sell_price = df.at[id, "bid"]
                    break

                turn += 1

            # Set result
            if sell_id is not None:
                df.at[buy_id, "buy_timestamp"] = df.at[buy_id, "timestamp"]
                df.at[buy_id, "buy_price"] = buy_price
                df.at[buy_id, "sell_timestamp"] = df.at[sell_id, "timestamp"]
                df.at[buy_id, "sell_price"] = sell_price
                df.at[buy_id, "profit"] = sell_price - buy_price
                df.at[buy_id, "profit_rate"] = df.at[buy_id, "profit"] / sell_price

        # Save
        L.info("Save data")

        app_s3.write_dataframe(df, f"{output_base_path}/fx.{fxpair}.{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
    except Exception as err:
        L.exception(f"fxpair={fxpair}, {err}")
        result["exception"] = err

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    execute(
        fxpairs=["usdjpy"],
        # years=[2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000],
        years=[2019, 2018, 2017],
        input_base_path=f"preprocess_1",
        output_base_path=f"preprocess_4",
    )
