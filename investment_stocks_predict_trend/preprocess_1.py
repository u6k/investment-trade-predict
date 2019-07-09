import pandas as pd
import numpy as np


def execute():
    input_base_path = "local/stock_prices"
    output_base_path = "local/preprocess_1"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0) \
        .sort_values("ticker_symbol") \
        .dropna() \
        .drop_duplicates() \
        .set_index("ticker_symbol")
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    for ticker_symbol in df_companies.index:
        print(f"ticker_symbol: {ticker_symbol}")

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        try:
            preprocess(ticker_symbol, input_base_path, output_base_path)
        except Exception as err:
            print(err)
            df_companies_result.at[ticker_symbol, "message"] = err.__str__()

        df_companies_result.to_csv(f"{output_base_path}/companies.csv")
        print(df_companies_result.loc[ticker_symbol])


def preprocess(ticker_symbol, input_base_path, output_base_path):
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


if __name__ == "__main__":
    execute()
