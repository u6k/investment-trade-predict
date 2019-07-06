import datetime
import joblib
import pandas as pd
import os


def preprocess():
    input_base_path = "local/backtest_preprocessed"
    output_base_path = "local/backtest_5"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    for ticker_symbol in df_companies.index:
        print(ticker_symbol)

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        try:
            # Load data
            clf = joblib.load(f"{input_base_path}/model.{ticker_symbol}.joblib")
            df_prices = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
            df_prices_all = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.all.csv", index_col=0)

            x = df_prices_all[[
                "volume_change_std",
                "adjusted_close_price_change_std",
                "sma_5_std",
                "sma_10_std",
                "sma_20_std",
                "sma_40_std",
                "sma_80_std",
                "momentum_5_std",
                "momentum_10_std",
                "momentum_20_std",
                "momentum_40_std",
                "momentum_80_std",
                "roc_5_std",
                "roc_10_std",
                "roc_20_std",
                "roc_40_std",
                "roc_80_std",
                "rsi_5_std",
                "rsi_10_std",
                "rsi_14_std",
                "rsi_20_std",
                "rsi_40_std",
                "stochastic_k_5_std",
                "stochastic_d_5_std",
                "stochastic_sd_5_std",
                "stochastic_k_9_std",
                "stochastic_d_9_std",
                "stochastic_sd_9_std",
                "stochastic_k_20_std",
                "stochastic_d_20_std",
                "stochastic_sd_20_std",
                "stochastic_k_25_std",
                "stochastic_d_25_std",
                "stochastic_sd_25_std",
                "stochastic_k_40_std",
                "stochastic_d_40_std",
                "stochastic_sd_40_std"
            ]].values

            # Predict
            y_pred = clf.predict(x)

            df_prices_all["predict"] = y_pred

            df_prices_predicted = df_prices[[
                "date",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume"
            ]].copy()
            for id in df_prices_all.index:
                df_prices_predicted.at[id, "predict"] = df_prices_all.at[id, "predict"]

            # Save
            df_prices_predicted.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.predicted.csv")

            os.remove(f"{input_base_path}/model.{ticker_symbol}.joblib")
            os.remove(f"{input_base_path}/stock_prices.{ticker_symbol}.csv")
            os.remove(f"{input_base_path}/stock_prices.{ticker_symbol}.all.csv")
        except Exception as err:
            print(err)
            df_companies_result.at[ticker_symbol, "message"] = err.__str__()

        print(df_companies_result.loc[ticker_symbol])

        df_companies_result.to_csv(f"{output_base_path}/companies.csv")


def backtest():
    base_path = "local/backtest_5"
    losscut_rate = 0.95
    risk_rate = 0.9
    total_risk_rate = 0.5

    # Load data
    df_companies = pd.read_csv(f"{base_path}/companies.csv", index_col=0)

    df_prices_dic = {}
    for ticker_symbol in df_companies.query("message.isnull() and score_0>0.6 and score_1>0.8 and score_1_total>50").index:
        print(f"load csv: {ticker_symbol}")
        df_prices_dic[ticker_symbol] = pd.read_csv(f"{base_path}/stock_prices.{ticker_symbol}.predicted.csv", index_col=0)

    # Backtest
    funds = 10000000
    assets = funds
    df_result = pd.DataFrame(columns=["buy_price", "buy_stocks", "losscut_price"])
    df_action = pd.DataFrame()
    df_stocks = pd.DataFrame(columns=["funds", "assets"])

    for date in date_array(2018):
        date_str = date.strftime("%Y-%m-%d")
        print(f"date={date_str}")

        # Buy
        df_buy_tmp = pd.DataFrame(columns=["open_price", "score_1"])
        for ticker_symbol in df_prices_dic.keys():
            print(f"  predict: {ticker_symbol}")
            df_prices_current = df_prices_dic[ticker_symbol].query(f"date=='{date_str}'")
            if len(df_prices_current) == 0:
                print("    no data")
                continue

            if df_prices_current["predict"].values[0] == 0:
                print("    stay")
                continue

            if ticker_symbol in df_stocks.index:
                print("    contain stocks")
                continue

            open_price = df_prices_current["open_price"].values[0]
            score_1 = df_companies.at[ticker_symbol, "score_1"]
            print(f"    predict 1: open_price={open_price}, score_1={score_1}")

            df_buy_tmp.at[ticker_symbol, "open_price"] = open_price
            df_buy_tmp.at[ticker_symbol, "score_1"] = score_1

        for ticker_symbol in df_buy_tmp.sort_values("score_1", ascending=False).index:
            print(f"  buy: {ticker_symbol}")

            buy_price = df_buy_tmp.at[ticker_symbol, "open_price"]
            buy_stocks = assets * (1 - risk_rate) // (buy_price * losscut_rate * 100) * 100

            if buy_stocks == 0:
                print("    buy stocks 0")
                continue

            if (assets * total_risk_rate > (funds - buy_price * buy_stocks)):
                print("    over total risk")
                continue

            print(f"    buy: buy_price={buy_price}, buy_stocks={buy_stocks}")
            df_stocks.at[ticker_symbol, "date"] = date_str
            df_stocks.at[ticker_symbol, "buy_price"] = buy_price
            df_stocks.at[ticker_symbol, "buy_stocks"] = buy_stocks
            df_stocks.at[ticker_symbol, "losscut_price"] = buy_price * losscut_rate

            action_id = len(df_action)
            df_action.at[action_id, "date"] = date_str
            df_action.at[action_id, "ticker_symbol"] = ticker_symbol
            df_action.at[action_id, "price"] = buy_price
            df_action.at[action_id, "losscut_price"] = buy_price * losscut_rate
            df_action.at[action_id, "stocks"] = buy_stocks

            funds -= buy_price * buy_stocks

        # Sell(losscut)
        for ticker_symbol in df_stocks.index:
            print(f"  sell: {ticker_symbol}")

            df_prices_current = df_prices_dic[ticker_symbol].query(f"date=='{date_str}'")
            if len(df_prices_current) == 0:
                print("    no data")
                continue

            losscut_price = df_stocks.at[ticker_symbol, "losscut_price"]
            low_price = df_prices_current["low_price"].values[0]

            if losscut_price < low_price:
                print("    hold")
                continue

            buy_stocks = df_stocks.at[ticker_symbol, "buy_stocks"]
            buy_price = df_stocks.at[ticker_symbol, "buy_price"]
            funds += low_price * buy_stocks

            df_stocks = df_stocks.drop(ticker_symbol)

            action_id = len(df_action)
            df_action.at[action_id, "date"] = date_str
            df_action.at[action_id, "ticker_symbol"] = ticker_symbol
            df_action.at[action_id, "price"] = low_price
            df_action.at[action_id, "stocks"] = -buy_stocks
            if buy_price < low_price:
                df_action.at[action_id, "result"] = "win"
            else:
                df_action.at[action_id, "result"] = "lose"

            print(f"     sell: sell_price={open_price}, sell_stocks={buy_stocks}")

        # Update losscut price
        for ticker_symbol in df_stocks.index:
            print(f"  update losscut price: {ticker_symbol}")

            df_prices_current = df_prices_dic[ticker_symbol].query(f"date=='{date_str}'")
            if len(df_prices_current) == 0:
                print("    no data")
                continue

            losscut_price = df_stocks.at[ticker_symbol, "losscut_price"]
            open_price = df_prices_current["open_price"].values[0]
            new_losscut_price = open_price * losscut_rate

            if new_losscut_price <= losscut_price:
                print("    no update")
                continue

            df_stocks.at[ticker_symbol, "losscut_price"] = new_losscut_price

            print(f"    update: old={losscut_price}, new={new_losscut_price}")

        # Turn end
        assets = funds

        for ticker_symbol in df_stocks.index:
            print(f"  update assets: {ticker_symbol}")

            df_prices_current = df_prices_dic[ticker_symbol].query(f"date=='{date_str}'")
            if len(df_prices_current) == 0:
                print("    no data")
                continue

            assets += df_prices_current["close_price"].values[0] * df_stocks.at[ticker_symbol, "buy_stocks"]

        df_result.at[date_str, "assets"] = assets
        df_result.at[date_str, "funds"] = funds

        df_result.to_csv(f"{base_path}/result.csv")
        df_action.to_csv(f"{base_path}/action.csv")
        df_stocks.to_csv(f"{base_path}/stocks.{date_str}.csv")

        print(df_result.loc[date_str])


def backtest_single():
    base_path = "local/backtest_5"
    year = 2018

    # Load data
    df_companies = pd.read_csv(f"{base_path}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    # Backtest
    for ticker_symbol in df_companies.index:
        print(ticker_symbol)

        try:
            df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

            df_prices = pd.read_csv(f"{base_path}/stock_prices.{ticker_symbol}.predicted.csv", index_col=0)
            df_result = backtest_single_impl(df_prices, year)
            df_companies_result.at[ticker_symbol, "total_profit"] = df_result["profit"].sum()
            df_companies_result.at[ticker_symbol, "message"] = ""

            # Save
            df_result.to_csv(f"{base_path}/backtest.{ticker_symbol}.csv")
        except Exception as err:
            print(err)
            df_companies_result.at[ticker_symbol, "message"] = err.__str__()

        print(df_companies_result.loc[ticker_symbol])

        df_companies_result.to_csv(f"{base_path}/result.csv")


def backtest_single_impl(df_prices, target_year):
    losscut_rate = 0.95

    df = df_prices.copy()

    # Backtest
    asset = 0
    buy_price = None
    losscut_price = None

    for date in date_array(target_year):
        date_str = date.strftime("%Y-%m-%d")

        # Skip
        df_current = df.query(f"date=='{date_str}'")
        if len(df_current) == 0:
            continue

        prices_id = df_current.index[0]

        # Buy
        if buy_price is None and df.at[prices_id-1, "predict"] == 1:
            buy_price = df.at[prices_id, "open_price"]
            losscut_price = buy_price * losscut_rate

            df.at[prices_id, "action"] = "buy"

        # Sell
        if losscut_price is not None and df.at[prices_id, "low_price"] < losscut_price:
            profit = df.at[prices_id, "low_price"] - buy_price
            asset += profit

            df.at[prices_id, "action"] = "sell"
            df.at[prices_id, "profit"] = profit

            buy_price = None
            losscut_price = None

        # Update losscut price
        if buy_price is not None:
            losscut_price_tmp = df.at[prices_id, "open_price"] * losscut_rate
            if losscut_price < losscut_price_tmp:
                losscut_price = losscut_price_tmp

        # Turn end
        df.at[prices_id, "asset"] = asset
        df.at[prices_id, "buy_price"] = buy_price
        df.at[prices_id, "losscut_price"] = losscut_price

    return df


def date_array(year):
    dates = []

    current_date = datetime.date(year, 1, 1)
    while current_date.year == year:
        dates.append(current_date)

        current_date += datetime.timedelta(days=1)

    return dates
