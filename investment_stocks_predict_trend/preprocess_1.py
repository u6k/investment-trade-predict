import joblib
import pandas as pd
import numpy as np

from app_logging import get_app_logger


def execute():
    L = get_app_logger(__name__)
    L.info("start")

    input_base_path = "local/stock_prices"
    output_base_path = "local/preprocess_1"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0) \
        .sort_values("ticker_symbol") \
        .dropna() \
        .drop_duplicates() \
        .set_index("ticker_symbol")
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(preprocess)(ticker_symbol, input_base_path, output_base_path) for ticker_symbol in df_companies.index])

    for result in results:
        ticker_symbol = result[0]
        message = result[1]

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]
        df_companies_result.at[ticker_symbol, "message"] = message

    df_companies_result.to_csv(f"{output_base_path}/companies.csv")
    L.info("finish")


def preprocess(ticker_symbol, input_base_path, output_base_path):
    L = get_app_logger(ticker_symbol)
    L.info(f"ticker_symbol: {ticker_symbol}")

    try:
        # Load data
        df = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        # Preprocess
        df = df.sort_values("date") \
            .dropna() \
            .drop_duplicates()
        df = df.assign(id=np.arange(len(df)))
        df = df.set_index("id")

        # Save data
        df.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.csv")

        message = ""
    except Exception as err:
        L.exception(err)
        message = err.__str__()

    return (ticker_symbol, message)


if __name__ == "__main__":
    execute()
