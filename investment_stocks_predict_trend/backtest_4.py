import datetime
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LassoCV, Ridge
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error



def execute():
    preprocess()


def preprocess():
    input_base_path = "local/backtest_preprocessed"
    output_base_path = "local/backtest_4"

    start_date = "2018-01-01"
    end_date = "2018-12-31"

    df_result = pd.read_csv(f"{input_base_path}/result.csv", index_col=0)
    df_companies = pd.DataFrame(columns=df_result.columns)

    for ticker_symbol in df_result.query("message.isnull()").index:
        print(ticker_symbol)

        try:
            # Load data
            clf = joblib.load(f"{input_base_path}/model.{ticker_symbol}.joblib")
            df_prices = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
            df_test_data = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.test_data.csv", index_col=0)

            x_test = []
            for index in df_prices.query(f"'{start_date}' <= date <= '{end_date}'").index:
                x_test.append(df_test_data.loc[index-1].values)

            # Predict
            y_pred = clf.predict(x_test)

            df_prices_predicted = df_prices.query(f"'{start_date}' <= date <= '{end_date}'")[["date", "open_price", "high_price", "low_price", "close_price", "volume", "adjusted_close_price"]]
            df_prices_predicted["predict"] = y_pred

            df_companies.loc[ticker_symbol] = df_result.loc[ticker_symbol]

            # Save
            df_prices_predicted.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.predicted.csv")
        except Exception as err:
            print(err)
            df_companies.at[ticker_symbol, "message"] = err.__str__()

        print(df_companies.loc[ticker_symbol])

        df_companies.to_csv(f"{output_base_path}/companies.csv")



def execute_old():
    base_path = "local/stock_prices_preprocessed"
    output_path = "local/test_3"

    df_companies = pd.read_csv(f"{base_path}/companies.csv")
    df_companies = df_companies.query("data_size > 2500")

    for ticker_symbol in df_companies["ticker_symbol"].values:
        df_prices = pd.read_csv(f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        for sma_len in [(5, 10), (5, 20), (5, 40), (5, 80), (10, 20), (10, 40), (10, 80), (20, 40), (20, 80), (40, 80)]:
            sma_short_len = sma_len[0]
            sma_long_len = sma_len[1]

            print(f"{sma_short_len}, {sma_long_len}, {ticker_symbol}")

            df_result = df_prices  # preprocess.simulate_trade(df_prices, sma_short_len, sma_long_len)
            df_result = df_result[["date",
                                   "open_price",
                                   "high_price",
                                   "low_price",
                                   "close_price",
                                   "adjusted_close_price",
                                   "adjusted_close_price_minmax",
                                   f"sma_{sma_short_len}",
                                   f"sma_{sma_long_len}",
                                   "buy_price",
                                   "losscut_price",
                                   "action",
                                   "profit"]]
            df_result = df_result.query("'2008-01-01' <= date <= '2018-12-31'")

            df_result.to_csv(f"{output_path}/result.{ticker_symbol}.{sma_short_len}_{sma_long_len}.csv")


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


def execute_2():
    load_base_dir = "local/stock_prices_preprocessed.old"
    test_base_dir = "local/test_3.2"

    sma_short_len = 5
    sma_long_len = 20
    losscut_rate = 0.95
    available_rate = 0.05
    total_available_rate = 0.5
    buy_stocks_unit = 100

    df_companies = pd.read_csv(f"{load_base_dir}/companies.csv", index_col=0) \
        .sort_values("ticker_symbol") \
        .set_index("ticker_symbol") \
        .query("data_size > 2500") \
        .query("volume_2018 > 100000000")
    print(df_companies.info())

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

                df_hold_stock = df_hold_stock.drop(ticker_symbol)
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

                df_hold_stock = df_hold_stock.drop(ticker_symbol)
                print(f"{ticker_symbol}: {action}")

            # update losscut price
            elif df_hold_stock.at[ticker_symbol, "losscut_price"] < (df_prices.at[current_id, "open_price"] * losscut_rate):
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
                # test_start_date = (date - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
                # test_end_date = date_str
                df_test_trade = pd.DataFrame()
                # df_test_trade = preprocess.simulate_trade(load_base_dir, ticker_symbol, test_start_date, test_end_date, sma_short_len, sma_long_len)
                df_test_trade = df_test_trade.dropna()
                last_assets = df_test_trade["assets"].values[-1]

                if last_assets > 20000000 and df_prices.at[current_id-3, "sma_80"] < df_prices.at[current_id-1, "sma_80"]:
                    df_buy_signal.at[ticker_symbol, "last_assets"] = last_assets
                    df_buy_signal.at[ticker_symbol, "open_price"] = df_prices.at[current_id, "open_price"]

        # 資産が高い銘柄順に、資金管理を行いつつ株を購入する
        if len(df_buy_signal) > 0:
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
