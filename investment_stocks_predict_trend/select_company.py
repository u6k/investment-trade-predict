import os
import pandas as pd
import numpy as np
import psycopg2


def export_stock_prices():
    con_config = {
        "host": os.environ["DB_HOST"],
        "port": os.environ["DB_PORT"],
        "database": os.environ["DB_DATABASE"],
        "user": os.environ["DB_USERNAME"],
        "password": os.environ["DB_PASSWORD"]
    }

    con = psycopg2.connect(**con_config)

    df_companies = pd.read_sql(sql="select * from companies", con=con)

    df_companies = df_companies[["ticker_symbol", "name", "market"]]
    df_companies = df_companies.sort_values("ticker_symbol")
    df_companies = df_companies.assign(id=np.arange(len(df_companies)))
    df_companies = df_companies.set_index("id")

    print(df_companies.info())

    for ticker_symbol in df_companies["ticker_symbol"].values:
        print(f"ticker_symbol: {ticker_symbol}")

        df = pd.read_sql(sql=f"select * from stock_prices where ticker_symbol='{ticker_symbol}'",
                         con=con)

        df = df.sort_values("date")
        df = df.assign(id=np.arange(len(df)))
        df = df.set_index("id")

        print(df.info())

        df.to_csv(f"local/stock_prices/stock_prices.{ticker_symbol}.csv")


def analysis():
    df_companies = pd.read_csv("local/companies.csv")

    df_companies = df_companies.dropna()
    df_companies = df_companies[["ticker_symbol", "name", "market"]]
    df_companies = df_companies.assign(ticker_symbol=df_companies["ticker_symbol"].astype(int))
    df_companies = df_companies.sort_values("ticker_symbol")
    df_companies = df_companies.assign(id=np.arange(len(df_companies)))
    df_companies = df_companies.set_index("id")

    print(df_companies.head())
    print(df_companies.info())

    df_analysed = df_companies.copy()

    for id in df_analysed.index:
        ticker_symbol = df_analysed.at[id, "ticker_symbol"]
        print(f"id: {id}, ticker_symbol: {ticker_symbol}")

        df_prices = pd.read_csv(f"local/stock_prices/stock_prices.{ticker_symbol}.csv", index_col=0)
        df_prices = df_prices.sort_values("date")
        df_prices = df_prices.drop_duplicates()
        df_prices["id"] = np.arange(len(df_prices))
        df_prices = df_prices.set_index("id")

        if len(df_prices) < 2500:
            continue

        for idx in df_prices.index:
            open_price = df_prices.at[idx, "open_price"]
            close_price = df_prices.at[idx, "close_price"]
            day_trade_profit = close_price - open_price

            if day_trade_profit > 0:
                df_prices.at[idx, "day_trade_profit"] = day_trade_profit
            else:
                df_prices.at[idx, "day_trade_profit"] = 0.0

        df_analysed.at[id, "data_size"] = len(df_prices)

        df_analysed.at[id, "latest_open_price"] = df_prices.at[df_prices.index[-1], "open_price"]
        df_analysed.at[id, "latest_high_price"] = df_prices.at[df_prices.index[-1], "high_price"]
        df_analysed.at[id, "latest_low_price"] = df_prices.at[df_prices.index[-1], "low_price"]
        df_analysed.at[id, "latest_close_price"] = df_prices.at[df_prices.index[-1], "close_price"]
        df_analysed.at[id, "latest_volume"] = df_prices.at[df_prices.index[-1], "volume"]
        df_analysed.at[id, "latest_adjusted_close_price"] = df_prices.at[df_prices.index[-1], "adjusted_close_price"]

        for window in [5, 10, 20, 40, 80]:
            df_prices[f"sma_{window}"] = df_prices["adjusted_close_price"].rolling(window).mean()

            sma_latest = df_prices.at[df_prices.index[-1], f"sma_{window}"]
            sma_start = df_prices.at[df_prices.index[-window], f"sma_{window}"]
            diff_sma = sma_latest - sma_start

            df_analysed.at[id, f"diff_sma_{window}"] = diff_sma

            df_analysed.at[id, f"volume_{window}"] = df_prices["volume"][-window:].values.sum()

            df_analysed.at[id, f"day_trade_profit_{window}"] = df_prices["day_trade_profit"][-window:].values.sum()

        for year in [2015, 2016, 2017, 2018, 2019]:
            df_analysed.at[id, f"start_id_{year}"] = df_prices.query(f"'{year}-01-01' <= date <= '{year}-12-31'").index[0]
            df_analysed.at[id, f"end_id_{year}"] = df_prices.query(f"'{year}-01-01' <= date <= '{year}-12-31'").index[-1]

        df_prices.to_csv(f"local/stock_prices/stock_prices.{ticker_symbol}.analysed.csv")
        df_analysed.to_csv("local/companies.analysed.csv")


def analysis_2():
    df = pd.read_csv("local/companies.analysed.csv", index_col=0)

    df = df[["ticker_symbol", "name", "data_size", "latest_open_price", "day_trade_profit_80"]]

    df = df.query("data_size > 2500")
    df = df.query("latest_open_price < 1000")
    df = df.sort_values("day_trade_profit_80", ascending=False)

    df.to_csv("local/companies.analysed.2.csv")
