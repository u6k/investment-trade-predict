import pandas as pd
import numpy as np
import os


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

        # Simple Moving Average
        for sma_len in [5, 10, 20, 40, 80]:
            df_prices[f"sma_{sma_len}"] = df_prices["adjusted_close_price"].rolling(sma_len).mean()

        # Momentum
        for momentum_len in [5, 10, 20, 40, 80]:
            df_prices[f"momentum_{momentum_len}"] = df_prices["adjusted_close_price"] - df_prices["adjusted_close_price"].shift(momentum_len-1)

        # ROC (Rate Of Change)
        for roc_len in [5, 10, 20, 40, 80]:
            df_prices[f"roc_{roc_len}"] = df_prices["adjusted_close_price"].pct_change(roc_len-1)

        # RSI
        for rsi_len in [5, 10, 14, 20, 40]:
            diff = df_prices["adjusted_close_price"].diff()
            diff = diff[1:]
            up, down = diff.copy(), diff.copy()
            up[up < 0] = 0
            down[down > 0] = 0
            up_sma = up.rolling(window=rsi_len, center=False).mean()
            down_sma = down.rolling(window=rsi_len, center=False).mean()
            rsi = up_sma / (up_sma - down_sma) * 100.0

            df_prices[f"rsi_{rsi_len}"] = rsi

        # Summary
        df_companies.at[ticker_symbol, "data_size"] = len(df_prices)
        for year in range(2008, 2019):
            df_companies.at[ticker_symbol, f"volume_{year}"] = df_prices.query(f"'{year}-01-01' <= date <= '{year}-12-31'")["volume"].sum()

        # Save
        df_prices.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        df_companies.to_csv(f"{output_base_path}/companies.csv")
