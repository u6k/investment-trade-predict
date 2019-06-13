import datetime
import pandas as pd

def execute():
    # target ticker_symbols
    df_companies = pd.read_csv("local/companies.analysed.csv", index_col=0)
    df_companies = df_companies.query("data_size > 0").copy()

    ticker_symbols = df_companies["ticker_symbol"].values

    # target dates
    dates = date_array(2018)

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        print(f"date: {date_str}")

        if calc_day_trade_profit(ticker_symbols[0], date_str) == None:
            continue

        df_profit = pd.DataFrame()
        for ticker_symbol in ticker_symbols:
            print(f"  ticker_symbol: {ticker_symbol}")

            profit = calc_day_trade_profit(ticker_symbol, date_str)
            df_profit.at[ticker_symbol, "profit"] = profit

        df_profit.to_csv(f"local/test_1/profit.{date_str}.csv")



def execute_single():
    ticker_symbol = "5610"
    year = 2018
    available_rate = 0.5
    buy_stocks_unit = 100
    funds = 10000000

    df = pd.read_csv(f"local/stock_prices/stock_prices.{ticker_symbol}.csv")
    dates = date_array(year)

    df_result = pd.DataFrame()
    df_result.at[dates[0], "funds"] = funds

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        print(date_str)

        df_current = df.query(f"date=='{date_str}'")
        if len(df_current) == 0:
            continue

        open_price = df_current["open_price"].values[0]
        close_price = df_current["close_price"].values[0]

        if close_price > open_price:
            buy_stocks = funds * available_rate // (open_price * buy_stocks_unit) * buy_stocks_unit
        else:
            buy_stocks = 0

        funds += -open_price * buy_stocks + close_price * buy_stocks

        df_result.at[date, "open_price"] = open_price
        df_result.at[date, "close_price"] = close_price
        df_result.at[date, "buy_stocks"] = buy_stocks
        df_result.at[date, "funds"] = funds

        df_result.to_csv(f"local/test_1/result.single.{year}.csv")




def calc_day_trade_profit(df, date):
    df_current = df.query(f"date=='{date}'").copy()
    if len(df_current) == 0:
        return None

    open_price = df_current["open_price"].values[0]
    close_price = df_current["close_price"].values[0]
    profit = close_price - open_price
    if profit < 0:
        profit = 0

    return profit




def date_array(year):
    dates = []

    current_date = datetime.date(year, 1, 1)
    while current_date.year == year:
        dates.append(current_date)

        current_date += datetime.timedelta(days=1)

    return dates
