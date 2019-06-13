import datetime
import pandas as pd


def execute():
    # target ticker_symbols
    df_companies = pd.read_csv("local/companies.analysed.csv", index_col=0)
    df_companies = df_companies.query("data_size > 2500")
    df_companies = df_companies.query("volume_80 > 100000000")

    ticker_symbols = df_companies["ticker_symbol"].values

    # read stock prices csv
    df_dic = {}
    for ticker_symbol in ticker_symbols:
        print(f"read csv: {ticker_symbol}")

        df = pd.read_csv(f"local/stock_prices/stock_prices.{ticker_symbol}.csv")
        df_dic[ticker_symbol] = df

    # other config
    year = 2018
    available_rate = 0.1
    total_available_rate = 0.5
    buy_stocks_unit = 100
    funds = 1000000

    # simulate
    dates = date_array(year)

    df_result = pd.DataFrame()
    df_result.at[dates[0], "funds"] = funds

    hold_stocks = []
    hold_count = 0

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        print(f"date: {date_str}")

        # calc profit
        df_profit = pd.DataFrame()
        for ticker_symbol in ticker_symbols:
            df = df_dic[ticker_symbol]

            df_current = df.query(f"date=='{date_str}'")
            if len(df_current) == 0:
                continue

            idx = df_current.index[-1]

            df_profit.at[ticker_symbol, "open_price"] = df.at[idx, "open_price"]
            df_profit.at[ticker_symbol, "close_price"] = df.at[idx, "close_price"]

            profit = 0
            for idx in df.query(f"date<'{date_str}'").index[-5:]:
                open_price = df.at[idx, "open_price"]
                close_price = df.at[idx, "close_price"]

                profit += close_price - open_price

            df_profit.at[ticker_symbol, "ticker_symbol"] = ticker_symbol
            df_profit.at[ticker_symbol, "profit"] = profit

        if len(df_profit) == 0:
            continue

        df_profit = df_profit.sort_values("profit")
        df_profit = df_profit.set_index("ticker_symbol")
        df_profit.to_csv(f"local/test_2/profit.{date_str}.csv")

        if len(hold_stocks) == 0:
            # buy
            funds_tmp = funds
            df_profit = df_profit.sort_values("profit")

            for ticker_symbol in df_profit.index:
                buy_stocks = funds * available_rate // (df_profit.at[ticker_symbol, "open_price"] * buy_stocks_unit) * buy_stocks_unit
                if buy_stocks == 0:
                    continue

                funds_tmp -= df_profit.at[ticker_symbol, "open_price"] * buy_stocks
                if funds_tmp < (funds * total_available_rate):
                    break

                hold_stocks.append({
                    "ticker_symbol": ticker_symbol,
                    "buy_price": df_profit.at[ticker_symbol, "open_price"],
                    "buy_stocks": buy_stocks
                })

            for hold_stock in hold_stocks:
                funds -= hold_stock["buy_price"] * hold_stock["buy_stocks"]

            hold_count = 1
        elif hold_count < 5:
            # stay
            hold_count += 1
        else:
            # sell
            for hold_stock in hold_stocks:
                funds += df_profit.at[hold_stock["ticker_symbol"], "close_price"] * hold_stock["buy_stocks"]

            hold_stocks = []
            hold_count = 0

        assets = funds
        for hold_stock in hold_stocks:
            assets += df_profit.at[hold_stock["ticker_symbol"], "close_price"] * hold_stock["buy_stocks"]

        df_result.at[date, "assets"] = assets
        df_result.at[date, "funds"] = funds

        for idx, hold_stock in enumerate(hold_stocks):
            df_result.at[date, f"ticker_symbol_{idx}"] = hold_stock["ticker_symbol"]
            df_result.at[date, f"buy_price_{idx}"] = hold_stock["buy_price"]
            df_result.at[date, f"buy_stocks_{idx}"] = hold_stock["buy_stocks"]

        df_result.to_csv(f"local/test_2/result.{year}.csv")


def date_array(year):
    dates = []

    current_date = datetime.date(year, 1, 1)
    while current_date.year == year:
        dates.append(current_date)

        current_date += datetime.timedelta(days=1)

    return dates
