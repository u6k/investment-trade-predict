import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def execute():
    input_base_path = "local/stock_prices_preprocessed_3"
    output_base_path = "local/stock_prices_preprocessed_4"

    train_start_date = "2008-01-01"
    train_end_date = "2017-12-31"
    test_start_date = "2018-01-01"
    test_end_date = "2018-12-31"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    for ticker_symbol in df_companies.index:
        print(ticker_symbol)

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        try:
            df_prices = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
            df_all, df_data_train, df_data_test, df_target_train, df_target_test = preprocess(
                df_prices,
                train_start_date,
                train_end_date,
                test_start_date,
                test_end_date
            )

            # Save
            df_all.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.all.csv")
            df_data_train.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.data_train.csv")
            df_data_test.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.data_test.csv")
            df_target_train.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.target_train.csv")
            df_target_test.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.target_test.csv")

            df_companies_result.at[ticker_symbol, "message"] = ""
        except Exception as err:
            print(err)
            df_companies_result.at[ticker_symbol, "message"] = f"error: {err.__str__()}"

        df_companies_result.to_csv(f"{output_base_path}/companies.csv")

        print(df_companies_result.loc[ticker_symbol])


def preprocess(df_prices, train_start_date, train_end_date, test_start_date, test_end_date):
    df = df_prices.copy()

    if len(df.query(f"'{train_start_date}' > date")) == 0 or len(df.query(f"date > '{test_end_date}'")) == 0:
        raise Exception("skip: little date")

    # drop columns
    df = df.drop(["ticker_symbol",
                  "open_price",
                  "high_price",
                  "low_price",
                  "close_price",
                  "trade_end_id",
                  "sell_price",
                  "profit",
                  "profit_rate"], axis=1)

    # simulate trade
    df["predict_target_label"] = df["day_trade_profit_flag"].shift(-1)

    df = df.drop(["day_trade_profit_rate", "day_trade_profit_flag"], axis=1)

    # volume
    volume_change = df["volume"] / df["volume"].shift(1)
    df["volume_change_std"] = StandardScaler().fit_transform(volume_change.values.reshape(-1, 1))

    df = df.drop("volume", axis=1)

    # adjusted close price
    adjusted_close_price_change = df["adjusted_close_price"] / df["adjusted_close_price"].shift(1)
    df["adjusted_close_price_change_std"] = StandardScaler().fit_transform(adjusted_close_price_change.values.reshape(-1, 1))

    df = df.drop("adjusted_close_price", axis=1)

    # SMA
    sma = []
    for sma_len in [5, 10, 20, 40, 80]:
        sma = np.append(sma, df[f"sma_{sma_len}"].values)

    scaler = StandardScaler().fit(sma.reshape(-1, 1))

    for sma_len in [5, 10, 20, 40, 80]:
        df[f"sma_{sma_len}_std"] = scaler.transform(df[f"sma_{sma_len}"].values.reshape(-1, 1))

        df = df.drop(f"sma_{sma_len}", axis=1)

    # Momentum
    momentum = []
    for momentum_len in [5, 10, 20, 40, 80]:
        momentum = np.append(momentum, df[f"momentum_{momentum_len}"].values)

    scaler = StandardScaler().fit(momentum.reshape(-1, 1))

    for momentum_len in [5, 10, 20, 40, 80]:
        df[f"momentum_{momentum_len}_std"] = scaler.transform(df[f"momentum_{momentum_len}"].values.reshape(-1, 1))

        df = df.drop(f"momentum_{momentum_len}", axis=1)

    # ROC
    roc = []
    for roc_len in [5, 10, 20, 40, 80]:
        roc = np.append(roc, df[f"roc_{roc_len}"].values)

    scaler = StandardScaler().fit(roc.reshape(-1, 1))

    for roc_len in [5, 10, 20, 40, 80]:
        df[f"roc_{roc_len}_std"] = scaler.transform(df[f"roc_{roc_len}"].values.reshape(-1, 1))

        df = df.drop(f"roc_{roc_len}", axis=1)

    # RSI
    rsi = []
    for rsi_len in [5, 10, 14, 20, 40]:
        rsi = np.append(rsi, df[f"rsi_{rsi_len}"].values)

    scaler = StandardScaler().fit(rsi.reshape(-1, 1))

    for rsi_len in [5, 10, 14, 20, 40]:
        df[f"rsi_{rsi_len}_std"] = scaler.transform(df[f"rsi_{rsi_len}"].values.reshape(-1, 1))

        df = df.drop(f"rsi_{rsi_len}", axis=1)

    # Stochastic
    stochastic = []
    for stochastic_len in [5, 9, 20, 25, 40]:
        stochastic = np.append(stochastic, df[f"stochastic_k_{stochastic_len}"].values)
        stochastic = np.append(stochastic, df[f"stochastic_d_{stochastic_len}"].values)
        stochastic = np.append(stochastic, df[f"stochastic_sd_{stochastic_len}"].values)

    scaler = StandardScaler().fit(stochastic.reshape(-1, 1))

    for stochastic_len in [5, 9, 20, 25, 40]:
        df[f"stochastic_k_{stochastic_len}_std"] = scaler.transform(df[f"stochastic_k_{stochastic_len}"].values.reshape(-1, 1))
        df[f"stochastic_d_{stochastic_len}_std"] = scaler.transform(df[f"stochastic_d_{stochastic_len}"].values.reshape(-1, 1))
        df[f"stochastic_sd_{stochastic_len}_std"] = scaler.transform(df[f"stochastic_sd_{stochastic_len}"].values.reshape(-1, 1))

        df = df.drop(f"stochastic_k_{stochastic_len}", axis=1)
        df = df.drop(f"stochastic_d_{stochastic_len}", axis=1)
        df = df.drop(f"stochastic_sd_{stochastic_len}", axis=1)

    # Split train and test
    df = df.dropna()

    train_start_id = df.query(f"'{train_start_date}' <= date <= '{train_end_date}'").index[0]
    train_end_id = df.query(f"'{train_start_date}' <= date <= '{train_end_date}'").index[-1]
    test_start_id = df.query(f"'{test_start_date}' <= date <= '{test_end_date}'").index[0]
    test_end_id = df.query(f"'{test_start_date}' <= date <= '{test_end_date}'").index[-1]

    df_data_train = df.loc[train_start_id: train_end_id].drop(["date", "predict_target_label"], axis=1)
    df_data_test = df.loc[test_start_id: test_end_id].drop(["date", "predict_target_label"], axis=1)
    df_target_train = df.loc[train_start_id: train_end_id][["predict_target_label"]]
    df_target_test = df.loc[test_start_id: test_end_id][["predict_target_label"]]

    return df, df_data_train, df_data_test, df_target_train, df_target_test
