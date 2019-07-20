import argparse
import joblib
import numpy as np
import pandas as pd

from app_logging import get_app_logger
import app_s3


def execute(*, s3_bucket, input_prices_base_path, input_indexes_base_path, output_base_path, test_mode):
    L = get_app_logger(__name__)
    L.info("start")

    # Prices

    # Load data
    df_companies = app_s3.read_dataframe(s3_bucket, f"{input_prices_base_path}/companies.csv", index_col=0) \
        .sort_values("ticker_symbol") \
        .dropna() \
        .drop_duplicates() \
        .set_index("ticker_symbol")
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    # Preprocess
    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(preprocess)(ticker_symbol, s3_bucket, input_prices_base_path, output_base_path, test_mode) for ticker_symbol in df_companies.index])

    # Total result
    for result in results:
        if result["exception"] is not None:
            continue

        ticker_symbol = result["ticker_symbol"]
        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

    # Indexes
    indexes = {
        "ni225": "日経平均株価",
        "topix": "TOPIX",
        "djia": "ダウ平均株価"
    }

    for ticker_symbol in indexes.keys():
        # Preprocess
        result = preprocess(ticker_symbol, s3_bucket, input_indexes_base_path, output_base_path, test_mode)

        if result["exception"] is not None:
            continue

        # Total result
        df_companies_result.at[ticker_symbol, "name"] = indexes[ticker_symbol]

    # Save data
    app_s3.write_dataframe(df_companies_result, s3_bucket, f"{output_base_path}/companies.csv")

    L.info("finish")


def preprocess(ticker_symbol, s3_bucket, input_base_path, output_base_path, test_mode):
    L = get_app_logger(f"preprocess_1.{ticker_symbol}")
    L.info(f"preprocess_1: {ticker_symbol}")

    result = {
        "ticker_symbol": ticker_symbol,
        "exception": None
    }

    try:
        if test_mode and type(ticker_symbol) is int and ticker_symbol > 1310:
            raise Exception("skip: test mode")

        # Load data
        df = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        # Preprocess
        df = df.sort_values("date") \
            .dropna() \
            .drop_duplicates()
        df = df.assign(id=np.arange(len(df)))
        df = df.set_index("id")

        if "ticker_symbol" not in df.columns:
            df["ticker_symbol"] = ticker_symbol
        if "volume" not in df.columns:
            df["volume"] = 0
        if "adjusted_close_price" not in df.columns:
            df["adjusted_close_price"] = df["close_price"]

        # Save data
        app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
    except Exception as err:
        L.exception(f"ticker_symbol={ticker_symbol}, {err}")
        result["exception"] = err

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", help="folder name suffix (default: test)", default="test")
    parser.add_argument("--test-mode", help="test mode (skip '> 1310')", default=False, type=bool)
    args = parser.parse_args()

    execute(
        s3_bucket="u6k",
        input_prices_base_path="ml-data/stocks/stock_prices",
        input_indexes_base_path="ml-data/stocks/stock_indexes",
        output_base_path=f"ml-data/stocks/preprocess_1.{args.suffix}",
        test_mode=args.test_mode
    )
