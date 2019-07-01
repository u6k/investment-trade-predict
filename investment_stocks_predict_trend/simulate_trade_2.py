import pandas as pd
import numpy as np


def execute():
    input_base_path = "local/stock_prices"
    output_base_path = "local/simulate_trade_2"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0) \
        .sort_values("ticker_symbol") \
        .drop_duplicates() \
        .set_index("ticker_symbol")
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    for ticker_symbol in df_companies.index:
        print(ticker_symbol)

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        df_prices = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
        df_prices = df_prices.sort_values("date")
        df_prices = df_prices.drop_duplicates()
        df_prices["id"] = np.arange(len(df_prices))
        df_prices = df_prices.set_index("id")

        df_result = simulate_trade(df_prices)

        df_result.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.simulated.csv")
        df_companies_result.to_csv(f"{output_base_path}/companies.csv")


def simulate_trade(df_prices, losscut_rate=0.95):
    df = df_prices.copy()

    # simulate
    for start_id in df.index:
        # reset
        losscut_price = df.at[start_id, "open_price"] * losscut_rate
        end_id = None
        current_id = start_id

        while df.index[-1] > current_id:
            # losscut
            if df.at[current_id, "low_price"] < losscut_price:
                end_id = current_id
                break

            # update losscut price
            if losscut_price < (df.at[current_id, "open_price"] * losscut_rate):
                losscut_price = df.at[current_id, "open_price"] * losscut_rate

            current_id += 1

        # set result
        if end_id is not None:
            df.at[start_id, "trade_end_id"] = end_id
            df.at[start_id, "sell_price"] = df.at[end_id, "low_price"]
            df.at[start_id, "profit"] = df.at[end_id, "low_price"] - df.at[start_id, "open_price"]
            df.at[start_id, "profit_rate"] = df.at[end_id, "low_price"] / df.at[start_id, "open_price"]

    return df
