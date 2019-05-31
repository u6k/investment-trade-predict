import pandas as pd
import numpy as np


def preprocessing():
    df_companies_csv = pd.read_csv("local/companies.csv")
    df_companies_csv.info()
    print(df_companies_csv.head())
    print(df_companies_csv.tail())

    df_profit = df_companies_csv.copy()
    df_profit = df_profit[["ticker_symbol", "name"]]
    df_profit = df_profit.dropna()
    df_profit = df_profit.assign(
        ticker_symbol=df_profit["ticker_symbol"].astype(int))
    df_profit = df_profit.sort_values("ticker_symbol")
    df_profit = df_profit.drop_duplicates()
    df_profit = df_profit.set_index("ticker_symbol")
    df_profit = df_profit.assign(profit=0.0)
    df_profit = df_profit.assign(volume=0.0)
    df_profit = df_profit.assign(data_size=0.0)

    df_profit.info()
    print(df_profit.head())
    print(df_profit.tail())

    for symbol in df_profit.index:
        df_prices_csv = pd.read_csv(
            "local/stock_prices." + str(symbol) + ".csv")

        df = df_prices_csv.copy()
        df = df[["date", "opening_price", "close_price", "turnover"]]
        df = df.sort_values("date")
        df = df.drop_duplicates()
        df = df.assign(id=np.arange(len(df)))
        df = df.set_index("id")

        for idx in df.index:
            if df.at[idx, "opening_price"] < df.at[idx, "close_price"]:
                df.at[idx, "profit"] = df.at[idx, "close_price"] - \
                    df.at[idx, "opening_price"]
            else:
                df.at[idx, "profit"] = 0.0

        df_profit.at[symbol, "data_size"] = len(df)

        if len(df) > 250:
            df_subset = df[-250:].copy()
            df_profit.at[symbol, "profit"] = df_subset["profit"].sum()
            df_profit.at[symbol, "volume"] = df_subset["turnover"].sum()

        print(df_profit.loc[symbol])

    df_profit.to_csv("local/profit.csv")

    return df_profit


def top():
    df_profit = pd.read_csv("local/profit.csv", index_col=0)
    df_profit.info()
    print(df_profit.head())
    print(df_profit.tail())

    df = df_profit.copy()
    df = df.query("data_size > 2500.0")
    df = df.query("volume > 10000000")
    df = df.query("profit > 5000.0")
    print(df)

    df.to_csv("local/top_companies.csv")

    return df
