import pandas as pd
import numpy as np
import os
import sklearn.preprocessing as sp


def preprocess():
    input_base_path = "local/stock_prices"
    output_base_path = "local/stock_prices_preprocessed"

    os.makedirs(output_base_path, exist_ok=True)

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies = df_companies.sort_values("ticker_symbol") \
        .drop_duplicates() \
        .set_index("ticker_symbol")

    for ticker_symbol in df_companies.index:
        print(f"ticker_symbol: {ticker_symbol}")

        df_prices = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0) \
            .sort_values("date") \
            .drop_duplicates()
        df_prices["id"] = np.arange(len(df_prices))
        df_prices = df_prices.set_index("id")

        # MinMax - adjusted close price
        df_prices["adjusted_close_price_minmax"] = sp.minmax_scale(df_prices["adjusted_close_price"])

        # Simple Moving Average
        for sma_len in [5, 10, 20, 40, 80]:
            df_prices[f"sma_{sma_len}"] = df_prices["adjusted_close_price_minmax"].rolling(sma_len).mean()

        # Day trade profit
        for prices_id in df_prices.index:
            day_trade_profit = df_prices.at[prices_id, "close_price"] - df_prices.at[prices_id, "open_price"]
            if day_trade_profit <= 0:
                day_trade_profit = 0.0
            df_prices.at[prices_id, "day_trade_profit"] = day_trade_profit

        # Summary
        df_companies.at[ticker_symbol, "data_size"] = len(df_prices)
        for year in range(2008, 2019):
            df_companies.at[ticker_symbol, f"volume_{year}"] = df_prices.query(f"'{year}-01-01' <= date <= '{year}-12-31'")["volume"].sum()
            df_companies.at[ticker_symbol, f"day_trade_profit_{year}"] = df_prices.query(f"'{year}-01-01' <= date <= '{year}-12-31'")["day_trade_profit"].sum()

        # Save
        df_prices.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        df_companies.to_csv(f"{output_base_path}/companies.csv")


def simulate_trade(df_prices_preprocessed, sma_short_len, sma_long_len, losscut_rate=0.95):
    df = df_prices_preprocessed.copy()

    # simulate
    buy_price = 0.0
    losscut_price = 0.0

    for current_id in df.index[sma_long_len:]:
        sma_short_1 = df.at[current_id-1, f"sma_{sma_short_len}"]
        sma_long_1 = df.at[current_id-1, f"sma_{sma_long_len}"]
        sma_short_2 = df.at[current_id-2, f"sma_{sma_short_len}"]
        sma_long_2 = df.at[current_id-2, f"sma_{sma_long_len}"]

        if (buy_price > 0) and (losscut_price > df.at[current_id-1, "close_price"]):
            # loss cut
            df.at[current_id, "profit"] = df.at[current_id, "close_price"] - buy_price
            df.at[current_id, "action"] = "losscut"

            buy_price = 0.0
            losscut_price = 0.0
        elif (buy_price > 0) and (sma_short_2 > sma_long_2) and (sma_short_1 <= sma_long_1):
            # sell
            df.at[current_id, "profit"] = df.at[current_id, "close_price"] - buy_price
            df.at[current_id, "action"] = "sell"

            buy_price = 0.0
            losscut_price = 0.0
        elif (buy_price == 0) and (sma_short_2 < sma_long_2) and (sma_short_1 >= sma_long_1):
            # buy
            df.at[current_id, "action"] = "buy"

            buy_price = df.at[current_id, "close_price"]
            losscut_price = buy_price * losscut_rate
        else:
            # stay
            df.at[current_id, "action"] = ""

            if losscut_price < (df.at[current_id, "close_price"] * losscut_rate):
                losscut_price = df.at[current_id, "close_price"] * losscut_rate

        df.at[current_id, "buy_price"] = buy_price
        df.at[current_id, "losscut_price"] = losscut_price

    return df
