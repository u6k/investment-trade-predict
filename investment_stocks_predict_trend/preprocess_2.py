import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def execute():
    input_base_path = "local/stock_prices_preprocessed"
    output_base_path = "local/predict_preprocessed"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies["message"] = "waiting"

    for ticker_symbol in df_companies.index:
        print(ticker_symbol)

        try:
            df_prices_preprocessed = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            if len(df_prices_preprocessed.query("'2013-01-01' > date")) == 0 or len(df_prices_preprocessed.query("date >= '2019-01-01'")) == 0:
                print("skip: little data")
                df_companies.at[ticker_symbol, "message"] = "skip: little data"
                continue

            df_prices = pd.DataFrame()
            df_prices["id"] = df_prices_preprocessed.index
            df_prices = df_prices.set_index("id")
            df_prices["date"] = df_prices_preprocessed["date"]

            # simulate trade
            df_prices["profit_rate"] = df_prices_preprocessed["profit_rate"]
            df_prices["profit_flag"] = df_prices_preprocessed["profit_rate"].apply(lambda r: 1 if r > 1.0 else 0)

            # volume
            volume_change = df_prices_preprocessed["volume"] / df_prices_preprocessed["volume"].shift(1)
            df_prices["volume_change_std"] = StandardScaler().fit_transform(volume_change.values.reshape(-1, 1))

            # adjusted close price
            adjusted_close_price_change = df_prices_preprocessed["adjusted_close_price"] / df_prices_preprocessed["adjusted_close_price"].shift(1)
            df_prices["adjusted_close_price_change_std"] = StandardScaler().fit_transform(adjusted_close_price_change.values.reshape(-1, 1))

            # SMA
            sma = []
            for sma_len in [5, 10, 20, 40, 80]:
                sma = np.append(sma, df_prices_preprocessed[f"sma_{sma_len}"].values)

            scaler = StandardScaler().fit(sma.reshape(-1, 1))

            for sma_len in [5, 10, 20, 40, 80]:
                df_prices[f"sma_{sma_len}_std"] = scaler.transform(df_prices_preprocessed[f"sma_{sma_len}"].values.reshape(-1, 1))

            # Momentum
            momentum = []
            for momentum_len in [5, 10, 20, 40, 80]:
                momentum = np.append(momentum, df_prices_preprocessed[f"momentum_{momentum_len}"].values)

            scaler = StandardScaler().fit(momentum.reshape(-1, 1))

            for momentum_len in [5, 10, 20, 40, 80]:
                df_prices[f"momentum_{momentum_len}_std"] = scaler.transform(df_prices_preprocessed[f"momentum_{momentum_len}"].values.reshape(-1, 1))

            # ROC
            roc = []
            for roc_len in [5, 10, 20, 40, 80]:
                roc = np.append(roc, df_prices_preprocessed[f"roc_{roc_len}"].values)

            scaler = StandardScaler().fit(roc.reshape(-1, 1))

            for roc_len in [5, 10, 20, 40, 80]:
                df_prices[f"roc_{roc_len}_std"] = scaler.transform(df_prices_preprocessed[f"roc_{roc_len}"].values.reshape(-1, 1))

            # RSI
            rsi = []
            for rsi_len in [5, 10, 14, 20, 40]:
                rsi = np.append(rsi, df_prices_preprocessed[f"rsi_{rsi_len}"].values)

            scaler = StandardScaler().fit(rsi.reshape(-1, 1))

            for rsi_len in [5, 10, 14, 20, 40]:
                df_prices[f"rsi_{rsi_len}_std"] = scaler.transform(df_prices_preprocessed[f"rsi_{rsi_len}"].values.reshape(-1, 1))

            # Stochastic
            stochastic = []
            for stochastic_len in [5, 9, 20, 25, 40]:
                stochastic = np.append(stochastic, df_prices_preprocessed[f"stochastic_k_{stochastic_len}"].values)
                stochastic = np.append(stochastic, df_prices_preprocessed[f"stochastic_d_{stochastic_len}"].values)
                stochastic = np.append(stochastic, df_prices_preprocessed[f"stochastic_sd_{stochastic_len}"].values)

            scaler = StandardScaler().fit(stochastic.reshape(-1, 1))

            for stochastic_len in [5, 9, 20, 25, 40]:
                df_prices[f"stochastic_k_{stochastic_len}_std"] = scaler.transform(
                    df_prices_preprocessed[f"stochastic_k_{stochastic_len}"].values.reshape(-1, 1))
                df_prices[f"stochastic_d_{stochastic_len}_std"] = scaler.transform(
                    df_prices_preprocessed[f"stochastic_d_{stochastic_len}"].values.reshape(-1, 1))
                df_prices[f"stochastic_sd_{stochastic_len}_std"] = scaler.transform(
                    df_prices_preprocessed[f"stochastic_sd_{stochastic_len}"].values.reshape(-1, 1))

            # Split train and test
            df_prices = df_prices.dropna()

            train_start_id = df_prices.query("'2013-01-01' <= date < '2018-01-01'").index[0]
            train_end_id = df_prices.query("'2013-01-01' <= date < '2018-01-01'").index[-1]
            test_start_id = df_prices.query("'2018-01-01' <= date < '2019-01-01'").index[0]
            test_end_id = df_prices.query("'2018-01-01' <= date < '2019-01-01'").index[-1]

            df_prices_train_data = df_prices.loc[train_start_id-1: train_end_id-1].drop(["date", "profit_rate", "profit_flag"], axis=1)
            df_prices_train_target = df_prices.loc[train_start_id: train_end_id][["profit_rate", "profit_flag"]]
            df_prices_test_data = df_prices.loc[test_start_id-1: test_end_id-1].drop(["date", "profit_rate", "profit_flag"], axis=1)
            df_prices_test_target = df_prices.loc[test_start_id: test_end_id][["profit_rate", "profit_flag"]]

            # Save
            df_prices_train_data.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.train_data.csv")
            df_prices_train_target.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.train_target.csv")
            df_prices_test_data.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.test_data.csv")
            df_prices_test_target.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.test_target.csv")

            df_companies.at[ticker_symbol, "message"] = ""
        except Exception as err:
            print(err)
            df_companies.at[ticker_symbol, "message"] = f"error: {err.__str__()}"

        df_companies.to_csv(f"{output_base_path}/companies.csv")
