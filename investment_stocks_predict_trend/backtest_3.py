import datetime
import pandas as pd
import numpy as np
import glob
import os

def execute():
    df_companies = pd.read_csv("local/companies.analysed.csv")
    df_companies = df_companies.query("data_size > 2500")
    df_companies = df_companies.query("volume_80 > 1000000")
    df_companies = df_companies.sort_values("ticker_symbol")

    for short_sma in [5]:
        for long_sma in [20]:
            for year in range(2018, 2007, -1):
                base_path = f"local/test_3/{short_sma}_{long_sma}/{year}"
                os.makedirs(base_path)

                df_companies.to_csv(f"{base_path}/companies.csv")

                for ticker_symbol in df_companies["ticker_symbol"].values:
                    print(f"ticker_symbol: {ticker_symbol}")
                    execute_single(base_path, ticker_symbol, year, short_sma, long_sma)

                execute_report(base_path)

def execute_report(base_path):
    df_report = pd.DataFrame()

    for file_path in glob.glob(f"{base_path}/result.*"):
        print(file_path)

        df = pd.read_csv(file_path)
        df_report.at[file_path, "assets"] = df["assets"].values[-1]
        df_report.at[file_path, "win"] = df["win"].values[-1]
        df_report.at[file_path, "lose"] = df["lose"].values[-1]
        df_report.at[file_path, "losscut"] = df["losscut"].values[-1]

    df_report = df_report.sort_values("assets", ascending=False)
    df_report.to_csv(f"{base_path}/report.csv")


def execute_report_2():
    df_report = pd.DataFrame()

    for year in range(2001, 2019):
        df_result = pd.read_csv(f"local/test_3.bak/5_20/{year}/report.csv")

        df_report.at[year, "max_assets"] = df_result["assets"].max()
        df_report.at[year, "total_assets"] = df_result["assets"].sum()
        df_report.at[year, "win"] = df_result["win"].sum()
        df_report.at[year, "lose"] = df_result["lose"].sum()

    df_report.to_csv("local/test_3/report_2.csv")
    



def execute_single(base_path, ticker_symbol, year, short_sma_len, long_sma_len):
    AVAILABLE_RATE = 0.5
    BUY_STOCK_UNIT = 100
    LOSS_CUT_RATE = 0.95

    # load data
    df_stocks = pd.read_csv(f"local/stock_prices/stock_prices.{ticker_symbol}.csv")
    df_stocks = df_stocks[["date", "open_price", "high_price", "low_price", "close_price", "volume", "adjusted_close_price"]]
    df_stocks = df_stocks.drop_duplicates()
    df_stocks = df_stocks.sort_values("date")
    df_stocks["id"] = np.arange(len(df_stocks))
    df_stocks = df_stocks.set_index("id")

    df_stocks[f"sma_{short_sma_len}"] = df_stocks["close_price"].rolling(short_sma_len).mean()
    df_stocks[f"sma_{long_sma_len}"] = df_stocks["close_price"].rolling(long_sma_len).mean()

    df_stocks.to_csv(f"{base_path}/stock_prices.{ticker_symbol}.csv")

    # simulate
    dates = date_array(year)

    funds = 10000000
    win = 0
    lose = 0
    losscut = 0
    buy_price = 0.0
    buy_stocks = 0

    df_result = pd.DataFrame()
    df_result.at[dates[0], "funds"] = funds
    df_result.at[dates[0], "assets"] = funds

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")

        df_current = df_stocks.query(f"date=='{date_str}'")
        if len(df_current) == 0:
            continue

        current_id = df_current.index[0]

        before_long_sma = df_stocks.at[current_id-2, f"sma_{long_sma_len}"]
        before_short_sma = df_stocks.at[current_id-2, f"sma_{short_sma_len}"]
        current_long_sma = df_stocks.at[current_id-1, f"sma_{long_sma_len}"]
        current_short_sma = df_stocks.at[current_id-1, f"sma_{short_sma_len}"]
        action = ""

        if (buy_stocks > 0) and ((buy_price * LOSS_CUT_RATE) > df_stocks.at[current_id, "close_price"]):
            # loss cut
            sell_price = df_stocks.at[current_id, "close_price"]
            funds += sell_price * buy_stocks

            buy_price = 0.0
            buy_stocks = 0

            losscut += 1
            lose += 1

            action = "loss cut"
        elif (buy_stocks == 0) and (before_long_sma > before_short_sma) and (current_long_sma <= current_short_sma):
            # buy
            buy_price = df_stocks.at[current_id, "close_price"]
            buy_stocks = (funds * AVAILABLE_RATE) // (buy_price * BUY_STOCK_UNIT) * BUY_STOCK_UNIT
            if buy_stocks > 0:
                funds -= buy_price * buy_stocks
                action = "buy"
            else:
                buy_price = 0.0
                buy_stocks = 0
        elif (buy_stocks > 0) and (before_long_sma < before_short_sma) and (current_long_sma >= current_short_sma):
            # sell
            sell_price = df_stocks.at[current_id, "close_price"]
            funds += sell_price * buy_stocks

            if buy_price < sell_price:
                win += 1
            else:
                lose += 1

            buy_price = 0.0
            buy_stocks = 0

            action = "sell"
        else:
            # stay
            pass

        df_result.at[date, f"sma_{short_sma_len}"] = df_stocks.at[current_id,  f"sma_{short_sma_len}"]
        df_result.at[date, f"sma_{long_sma_len}"] = df_stocks.at[current_id,  f"sma_{long_sma_len}"]
        df_result.at[date, "close_price"] = df_stocks.at[current_id,  "close_price"]

        df_result.at[date, "funds"] = funds
        df_result.at[date, "assets"] = funds + buy_stocks * df_stocks.at[current_id, "close_price"]
        df_result.at[date, "buy_price"] = buy_price
        df_result.at[date, "buy_stocks"] = buy_stocks
        df_result.at[date, "win"] = win
        df_result.at[date, "lose"] = lose
        df_result.at[date, "losscut"] = losscut
        df_result.at[date, "action"] = action

        df_result.to_csv(f"{base_path}/result.{ticker_symbol}.csv")



def date_array(year):
    dates = []

    current_date = datetime.date(year, 1, 1)
    while current_date.year == year:
        dates.append(current_date)

        current_date += datetime.timedelta(days=1)

    return dates
