import datetime
import pandas as pd
import numpy as np
import os
import sklearn.preprocessing as sp
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LassoCV, Ridge
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error


def preprocess():
    df_companies = pd.read_csv("local/stock_prices/companies.csv", index_col=0) \
        .sort_values("ticker_symbol")
    df_companies["id"] = np.arange(len(df_companies))
    df_companies = df_companies.set_index("id")

    for ticker_symbol in df_companies["ticker_symbol"]:
        print(f"ticker_symbol: {ticker_symbol}")

        df_prices = pd.read_csv(f"local/stock_prices/stock_prices.{ticker_symbol}.csv", index_col=0) \
            .sort_values("date") \
            .drop_duplicates()

        df_prices["id"] = np.arange(len(df_prices))
        df_prices = df_prices.set_index("id")

        df_prices["adjusted_close_price_minmax"] = sp.minmax_scale(df_prices["adjusted_close_price"])

        for sma_len in [5, 10, 20, 40, 80]:
            df_prices[f"sma_{sma_len}"] = df_prices["adjusted_close_price_minmax"].rolling(sma_len).mean()

        for rsi_len in [5, 10, 14, 20, 40]:
            for prices_id in df_prices.index[rsi_len:]:
                up_total = 0.0
                down_total = 0.0
                for i in range(0, rsi_len):
                    diff = df_prices.at[prices_id-i, "adjusted_close_price"] - df_prices.at[prices_id-rsi_len, "adjusted_close_price"]
                    if diff >= 0.0:
                        up_total += diff
                    else:
                        down_total -= diff
                if up_total > 0.0:
                    df_prices.at[prices_id, f"rsi_{rsi_len}"] = up_total / (up_total + down_total) * 100.0
                else:
                    df_prices.at[prices_id, f"rsi_{rsi_len}"] = 0.0

        for roc_len in [5, 10, 20, 25, 40]:
            for prices_id in df_prices.index[roc_len:]:
                df_prices.at[prices_id, f"roc_{roc_len}"] = df_prices.at[prices_id, "adjusted_close_price"] / \
                    df_prices.at[prices_id-roc_len, "adjusted_close_price"] * 100.0

        for momentum_len in [5, 10, 20, 40]:
            for prices_id in df_prices.index[momentum_len:]:
                df_prices.at[prices_id, f"momentum_{momentum_len}"] = \
                    df_prices.at[prices_id, "adjusted_close_price"] \
                    - df_prices.at[prices_id-momentum_len, "adjusted_close_price"]

        df_prices["return_rate"] = df_prices["close_price"] / df_prices["open_price"]

        for prices_id in df_prices.index:
            day_trade_profit = df_prices.at[prices_id, "close_price"] - df_prices.at[prices_id, "open_price"]
            if day_trade_profit <= 0:
                day_trade_profit = 0.0
            df_prices.at[prices_id, "day_trade_profit"] = day_trade_profit

        companies_id = df_companies.query(f"ticker_symbol=='{ticker_symbol}'").index[0]
        df_companies.at[companies_id, "data_size"] = len(df_prices)
        for year in range(2008, 2019):
            df_companies.at[companies_id, f"volume_{year}"] = df_prices.query(f"'{year}-01-01' <= date <= '{year}-12-31'")["volume"].sum()
            df_companies.at[companies_id, f"day_trade_profit_{year}"] = df_prices.query(f"'{year}-01-01' <= date <= '{year}-12-31'")["day_trade_profit"].sum()

        df_prices.to_csv(f"local/stock_prices_preprocessed/stock_prices.{ticker_symbol}.csv")
        df_companies.to_csv("local/stock_prices_preprocessed/companies.csv")


def execute():
    df_companies = pd.read_csv("local/stock_prices_preprocessed/companies.csv")
    df_companies = df_companies.query("data_size > 2500")
    df_companies = df_companies.query("volume_2018 > 1000000")
    df_companies = df_companies.sort_values("ticker_symbol")

    for short_sma in [5]:
        for long_sma in [20]:
            for year in range(2018, 2007, -1):
                base_path = f"local/test_3/{short_sma}_{long_sma}/{year}"
                os.makedirs(base_path)

                df_companies.to_csv(f"{base_path}/companies.csv")

                for ticker_symbol in df_companies["ticker_symbol"].values:
                    print(f"ticker_symbol: {ticker_symbol}")
                    df_stocks = simulate_trade(ticker_symbol, f"{year}-01-01", f"{year}-12-31", short_sma, long_sma)

                    df_stocks.to_csv(f"{base_path}/result.{ticker_symbol}.csv")


def report():
    year = 2010
    base_path = f"local/test_3/5_20/{year}"

    df_report = pd.DataFrame()

    df_companies = pd.read_csv(f"{base_path}/companies.csv") \
        .set_index("id")

    for ticker_symbol in df_companies["ticker_symbol"].values:
        print(f"ticker_symbol: {ticker_symbol}")

        df = pd.read_csv(f"{base_path}/result.{ticker_symbol}.csv") \
            .query(f"'{year}-01-01' <= date <= '{year}-12-31'")

        df_report.at[ticker_symbol, "assets"] = df["assets"].values[-1]
        df_report.at[ticker_symbol, "win"] = int(df["win"].values[-1])
        df_report.at[ticker_symbol, "lose"] = int(df["lose"].values[-1])
        df_report.at[ticker_symbol, "losscut"] = int(df["losscut"].values[-1])

    df_report = df_report.sort_values("assets", ascending=False)

    df_report.to_csv(f"{base_path}/report.csv")


def report_2():
    df_report = pd.DataFrame()

    short_sma_len = 5
    long_sma_len = 20

    for year in range(2018, 2009, -1):
        df = pd.read_csv(f"local/test_3/report.{short_sma_len}_{long_sma_len}.{year}.csv", index_col=0)

        df_report.at[year, "max_assets"] = df["assets"].max()
        df_report.at[year, "total_win"] = df["win"].sum()
        df_report.at[year, "total_lose"] = df["lose"].sum()
        df_report.at[year, "total_losscut"] = df["losscut"].sum()

    df_report.to_csv(f"local/test_3/report_2.csv")


def train_profit_rate():
    df_report = pd.read_csv("local/test_3/5_20/2018/report_2.csv", index_col=0)

    x, y = [], []
    for index in df_report.index:
        x_data = []
        for i in range(1, 20):
            x_data.append(df_report.at[index, f"sma_5_{i}"])
        for i in range(1, 20):
            x_data.append(df_report.at[index, f"sma_20_{i}"])
        x.append(x_data)

        # y.append(df_report.at[index, "profit_rate"])
        if df_report.at[index, "profit_rate"] < 1.0:
            y.append(0)
        else:
            y.append(1)

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # clf = LassoCV()
    # clf = Ridge()
    clf = SVC(kernel="linear", C=1.)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    print(mse)

    df_test = pd.DataFrame()
    df_test["pred"] = y_pred
    df_test["test"] = y_test
    for index in df_test.index:
        df_test.at[index, "result"] = (df_test.at[index, "pred"] >= 1.0 and df_test.at[index, "test"] >= 1.0) \
            or (df_test.at[index, "pred"] < 1.0 and df_test.at[index, "test"] < 1.0)
    df_test.to_csv("local/test_3/5_20/2018/test.csv")


def simulate_trade(base_path, ticker_symbol, start_date, end_date, short_sma_len, long_sma_len):
    AVAILABLE_RATE = 0.5
    BUY_STOCK_UNIT = 100
    LOSS_CUT_RATE = 0.95

    # load data
    df_stocks = pd.read_csv(f"{base_path}/stock_prices.{ticker_symbol}.csv")
    df_stocks = df_stocks[["date", "open_price", "high_price", "low_price", "close_price", "volume",
                           "adjusted_close_price", "adjusted_close_price_minmax", f"sma_{short_sma_len}", f"sma_{long_sma_len}"]]
    df_stocks = df_stocks.drop_duplicates()
    df_stocks = df_stocks.sort_values("date")
    df_stocks["id"] = np.arange(len(df_stocks))
    df_stocks = df_stocks.set_index("id")

    # simulate
    df_tmp = df_stocks.query(f"'{start_date}' <= date <= '{end_date}'")
    start_id = df_tmp.index[0]
    end_id = df_tmp.index[-1]
    current_id = start_id

    funds = 10000000
    win = 0
    lose = 0
    losscut = 0
    buy_price = 0.0
    buy_stocks = 0
    losscut_price = 0.0
    profit_take_price = 0.0

    while current_id <= end_id:
        before_long_sma = df_stocks.at[current_id-2, f"sma_{long_sma_len}"]
        before_short_sma = df_stocks.at[current_id-2, f"sma_{short_sma_len}"]
        current_long_sma = df_stocks.at[current_id-1, f"sma_{long_sma_len}"]
        current_short_sma = df_stocks.at[current_id-1, f"sma_{short_sma_len}"]
        action = ""

        if (buy_stocks > 0) and (losscut_price > df_stocks.at[current_id, "close_price"]):
            # loss cut
            sell_price = df_stocks.at[current_id, "close_price"]
            funds += sell_price * buy_stocks

            buy_price = 0.0
            buy_stocks = 0

            losscut += 1
            lose += 1

            action = "losscut"
        elif (buy_stocks > 0) and (profit_take_price > df_stocks.at[current_id, "close_price"]):
            # profit take
            sell_price = df_stocks.at[current_id, "close_price"]
            funds += sell_price * buy_stocks

            buy_price = 0.0
            buy_stocks = 0

            win += 1

            action = "sell"
        elif (buy_stocks == 0) and (before_long_sma > before_short_sma) and (current_long_sma <= current_short_sma):
            # buy
            buy_price = df_stocks.at[current_id, "close_price"]
            buy_stocks = (funds * AVAILABLE_RATE) // (buy_price * BUY_STOCK_UNIT) * BUY_STOCK_UNIT
            if buy_stocks > 0:
                funds -= buy_price * buy_stocks
                losscut_price = buy_price * LOSS_CUT_RATE
                profit_take_price = buy_price * LOSS_CUT_RATE
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
            if profit_take_price < (df_stocks.at[current_id, "close_price"] * LOSS_CUT_RATE):
                profit_take_price = df_stocks.at[current_id, "close_price"] * LOSS_CUT_RATE

        df_stocks.at[current_id, "funds"] = funds
        df_stocks.at[current_id, "assets"] = funds + buy_stocks * df_stocks.at[current_id, "close_price"]
        df_stocks.at[current_id, "buy_price"] = buy_price
        df_stocks.at[current_id, "buy_stocks"] = buy_stocks
        df_stocks.at[current_id, "win"] = win
        df_stocks.at[current_id, "lose"] = lose
        df_stocks.at[current_id, "losscut"] = losscut
        df_stocks.at[current_id, "action"] = action

        current_id += 1

    return df_stocks.copy()


def execute_2():
    load_base_dir = "local/stock_prices_preprocessed.old"
    test_base_dir = "local/test_3.2"

    sma_short_len = 5
    sma_long_len = 20
    losscut_rate = 0.95
    available_rate = 0.1
    total_available_rate = 0.5
    buy_stocks_unit = 100

    df_companies = pd.read_csv(f"{load_base_dir}/companies.csv", index_col=0) \
        .sort_values("ticker_symbol") \
        .set_index("ticker_symbol") \
        .query("data_size > 2500") \
        .query("volume_2018 > 10000000")

    funds = 10000000
    win = 0
    lose = 0

    df_result = pd.DataFrame()
    df_action = pd.DataFrame()
    df_hold_stock = pd.DataFrame()

    for date in date_array(2018):
        date_str = date.strftime("%Y-%m-%d")
        print(f"*** date={date_str} ***")

        # skip
        df_prices = pd.read_csv(f"{load_base_dir}/stock_prices.{df_companies.index[0]}.csv", index_col=0)
        if len(df_prices.query(f"date=='{date_str}'")) == 0:
            print("skip")
            continue

        for ticker_symbol in df_hold_stock.index:
            df_prices = pd.read_csv(f"{load_base_dir}/stock_prices.{ticker_symbol}.csv", index_col=0)

            current_id = df_prices.query(f"date=='{date_str}'").index[0]
            sma_short_1 = df_prices.at[current_id-1, f"sma_{sma_short_len}"]
            sma_long_1 = df_prices.at[current_id-1, f"sma_{sma_long_len}"]
            sma_short_2 = df_prices.at[current_id-2, f"sma_{sma_short_len}"]
            sma_long_2 = df_prices.at[current_id-2, f"sma_{sma_long_len}"]

            # losscut
            if df_prices.at[current_id-1, "close_price"] < df_hold_stock.at[ticker_symbol, "losscut_price"]:
                sell_price = df_prices.at[current_id, "open_price"]
                funds += sell_price * df_hold_stock.at[ticker_symbol, "buy_stocks"]

                if sell_price > df_hold_stock.at[ticker_symbol, "buy_price"]:
                    win += 1
                    action = "losscut win"
                else:
                    lose += 1
                    action = "losscut lose"

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = action
                df_action.at[action_id, "price"] = sell_price
                df_action.at[action_id, "stocks"] = df_hold_stock.at[ticker_symbol, "buy_stocks"]

                df_hold_stock.drop(ticker_symbol)
                print(f"{ticker_symbol}: {action}")

            # sell signal
            elif (sma_short_2 > sma_long_2) and (sma_short_1 <= sma_long_1):
                sell_price = df_prices.at[current_id, "open_price"]
                funds += sell_price * df_hold_stock.at[ticker_symbol, "buy_stocks"]

                if sell_price > df_hold_stock.at[ticker_symbol, "buy_price"]:
                    win += 1
                    action = "sell win"
                else:
                    lose += 1
                    action = "sell lose"

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = action
                df_action.at[action_id, "price"] = sell_price
                df_action.at[action_id, "stocks"] = df_hold_stock.at[ticker_symbol, "buy_stocks"]

                df_hold_stock.drop(ticker_symbol)
                print(f"{ticker_symbol}: {action}")

            # update losscut price
            if df_hold_stock.at[ticker_symbol, "losscut_price"] < (df_prices.at[current_id, "open_price"] * losscut_rate):
                df_hold_stock.at[ticker_symbol, "losscut_price"] = df_prices.at[current_id, "open_price"] * losscut_rate

        # buy signal
        df_buy_signal = pd.DataFrame()

        for ticker_symbol in df_companies.index:
            df_prices = pd.read_csv(f"{load_base_dir}/stock_prices.{ticker_symbol}.csv", index_col=0)

            current_id = df_prices.query(f"date=='{date_str}'").index[0]

            sma_short_1 = df_prices.at[current_id-1, f"sma_{sma_short_len}"]
            sma_long_1 = df_prices.at[current_id-1, f"sma_{sma_long_len}"]
            sma_short_2 = df_prices.at[current_id-2, f"sma_{sma_short_len}"]
            sma_long_2 = df_prices.at[current_id-2, f"sma_{sma_long_len}"]

            # 買いシグナルが点灯した銘柄について、過去1年間の移動平均法売買における資産を算出
            if (sma_short_2 < sma_long_2) and (sma_short_1 >= sma_long_1):
                test_start_date = (date - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
                test_end_date = date_str
                df_test_trade = simulate_trade(load_base_dir, ticker_symbol, test_start_date, test_end_date, sma_short_len, sma_long_len)
                df_test_trade = df_test_trade.dropna()
                last_assets = df_test_trade["assets"].values[-1]

                if last_assets > 10000000:
                    df_buy_signal.at[ticker_symbol, "last_assets"] = last_assets
                    df_buy_signal.at[ticker_symbol, "open_price"] = df_prices.at[current_id, "open_price"]

        # 資産が高い銘柄順に、資金管理を行いつつ株を購入する
        base_funds = funds
        for ticker_symbol in df_hold_stock.index:
            base_funds += df_hold_stock.at[ticker_symbol, "buy_price"] * df_hold_stock.at[ticker_symbol, "buy_stocks"]

        for ticker_symbol in df_buy_signal.sort_values("last_assets", ascending=False).index:
            buy_price = df_buy_signal.at[ticker_symbol, "open_price"]
            buy_stocks = (base_funds * available_rate) // (buy_price * buy_stocks_unit) * buy_stocks_unit
            if (base_funds * total_available_rate) > (funds - buy_price * buy_stocks):
                buy_stocks = 0

            action_id = len(df_action)
            df_action.at[action_id, "date"] = date
            df_action.at[action_id, "ticker_symbol"] = ticker_symbol
            df_action.at[action_id, "action"] = "buy"
            df_action.at[action_id, "price"] = buy_price
            df_action.at[action_id, "stocks"] = buy_stocks

            if buy_stocks > 0:
                df_hold_stock.at[ticker_symbol, "date"] = date
                df_hold_stock.at[ticker_symbol, "ticker_symbol"] = ticker_symbol
                df_hold_stock.at[ticker_symbol, "buy_price"] = buy_price
                df_hold_stock.at[ticker_symbol, "buy_stocks"] = buy_stocks
                df_hold_stock.at[ticker_symbol, "losscut_price"] = buy_price * losscut_rate

                funds -= buy_price * buy_stocks

            print(f"{ticker_symbol}: buy")

        # turn end
        result_id = len(df_result)
        df_result.at[result_id, "funds"] = funds
        df_result.at[result_id, "win"] = win
        df_result.at[result_id, "lose"] = lose

        df_result.at[result_id, "assets"] = funds
        for ticker_symbol in df_hold_stock.index:
            df_prices = pd.read_csv(f"{load_base_dir}/stock_prices.{ticker_symbol}.csv", index_col=0)
            current_id = df_prices.query(f"date=='{date_str}'").index[0]

            df_result.at[result_id, "assets"] += df_prices.at[current_id, "close_price"] * df_hold_stock.at[ticker_symbol, "buy_stocks"]

        df_result.to_csv(f"{test_base_dir}/result.csv")
        df_action.to_csv(f"{test_base_dir}/action.csv")
        df_hold_stock.to_csv(f"{test_base_dir}/hold_stock.{date_str}.csv")

        print(df_result.loc[result_id])


def date_array(year):
    dates = []

    current_date = datetime.date(year, 1, 1)
    while current_date.year == year:
        dates.append(current_date)

        current_date += datetime.timedelta(days=1)

    return dates
