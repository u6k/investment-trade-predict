import pandas as pd
import numpy as np


VERSION = '0.2.0-develop'



def hello():
    return "hello"

def processing_by_company():
    df_companies_csv = pd.read_csv("local/companies.csv")
    df_companies_csv.info()
    print(df_companies_csv.head())
    print(df_companies_csv.tail())

    df_result = df_companies_csv.copy()
    df_result = df_result[["ticker_symbol", "name"]]
    df_result = df_result.dropna()
    df_result = df_result.assign(ticker_symbol=df_result["ticker_symbol"].astype(int))
    df_result = df_result.sort_values("ticker_symbol")
    df_result = df_result.drop_duplicates()
    df_result = df_result.set_index("ticker_symbol")
    df_result = df_result.assign(profit=0.0)
    df_result = df_result.assign(volume=0.0)
    df_result = df_result.assign(data_size=0.0)

    df_result.info()
    print(df_result.head())
    print(df_result.tail())

    for symbol in df_result.index:
        df_prices_csv = pd.read_csv("local/stock_prices." + str(symbol) + ".csv")

        df = df_prices_csv.copy()
        df = df[["date", "opening_price", "close_price", "turnover"]]
        df = df.sort_values("date")
        df = df.drop_duplicates()
        df = df.assign(id=np.arange(len(df)))
        df = df.set_index("id")

        for idx in df.index:
            if df.at[idx, "opening_price"] < df.at[idx, "close_price"]:
                df.at[idx, "profit"] = df.at[idx, "close_price"] - df.at[idx, "opening_price"]
            else:
                df.at[idx, "profit"] = 0.0

        df_result.at[symbol, "data_size"] = len(df)

        if len(df) > 250:
            df_subset = df[-250:].copy()
            df_result.at[symbol, "profit"] = df_subset["profit"].sum()
            df_result.at[symbol, "volume"] = df_subset["turnover"].sum()

        print(df_result.loc[symbol])

    df_result.to_csv("local/result.csv")

    return df_result

