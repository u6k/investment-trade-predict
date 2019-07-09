import pandas as pd


def execute():
    input_base_path_preprocess = "local/preprocess_2"
    input_base_path_simulate = "local/simulate_trade_2"
    output_base_path = "local/preprocess_3"

    train_start_date = "2008-01-01"
    train_end_date = "2017-12-31"
    test_start_date = "2018-01-01"
    test_end_date = "2018-12-31"

    df_companies = pd.read_csv(f"{input_base_path_preprocess}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    for ticker_symbol in df_companies.index:
        print(f"ticker_symbol={ticker_symbol}")

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        try:
            preprocess(ticker_symbol, input_base_path_preprocess, input_base_path_simulate, output_base_path)
            train_test_split(ticker_symbol, output_base_path, train_start_date, train_end_date, test_start_date, test_end_date)

            df_companies_result.at[ticker_symbol, "message"] = ""
        except Exception as err:
            df_companies_result.at[ticker_symbol, "message"] = err.__str__()

        df_companies_result.to_csv(f"{output_base_path}/companies.csv")
        print(df_companies_result.loc[ticker_symbol])


def preprocess(ticker_symbol, input_base_path_preprocess, input_base_path_simulate, output_base_path):
    df_preprocess = pd.read_csv(f"{input_base_path_preprocess}/stock_prices.{ticker_symbol}.csv", index_col=0)
    df_simulate = pd.read_csv(f"{input_base_path_simulate}/stock_prices.{ticker_symbol}.csv", index_col=0)

    df = df_preprocess.drop([
        "ticker_symbol",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "adjusted_close_price",
        "volume_change",
        "adjusted_close_price_change",
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_40",
        "sma_80",
        "momentum_5",
        "momentum_10",
        "momentum_20",
        "momentum_40",
        "momentum_80",
        "roc_5",
        "roc_10",
        "roc_20",
        "roc_40",
        "roc_80",
        "rsi_5",
        "rsi_10",
        "rsi_14",
        "rsi_20",
        "rsi_40",
        "stochastic_k_5",
        "stochastic_d_5",
        "stochastic_sd_5",
        "stochastic_k_9",
        "stochastic_d_9",
        "stochastic_sd_9",
        "stochastic_k_20",
        "stochastic_d_20",
        "stochastic_sd_20",
        "stochastic_k_25",
        "stochastic_d_25",
        "stochastic_sd_25",
        "stochastic_k_40",
        "stochastic_d_40",
        "stochastic_sd_40"
    ], axis=1).copy()

    df["predict_target_value"] = df_simulate["profit_rate"].shift(-1)
    df["predict_target_label"] = df["predict_target_value"].apply(lambda v: 1 if v > 1.0 else 0)

    df.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.csv")


def train_test_split(ticker_symbol, base_path, train_start_date, train_end_date, test_start_date, test_end_date):
    df = pd.read_csv(f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0) \
        .dropna()

    if len(df.query(f"date < '{train_start_date}'")) == 0 or len(df.query(f"date > '{test_end_date}'")) == 0:
        raise Exception("little data")

    train_start_id = df.query(f"'{train_start_date}' <= date <= '{train_end_date}'").index[0]
    train_end_id = df.query(f"'{train_start_date}' <= date <= '{train_end_date}'").index[-1]
    test_start_id = df.query(f"'{test_start_date}' <= date <= '{test_end_date}'").index[0]
    test_end_id = df.query(f"'{test_start_date}' <= date <= '{test_end_date}'").index[-1]

    df_data_train = df.loc[train_start_id: train_end_id].drop(["date", "predict_target_value", "predict_target_label"], axis=1)
    df_data_test = df.loc[test_start_id: test_end_id].drop(["date", "predict_target_value", "predict_target_label"], axis=1)
    df_target_train = df.loc[train_start_id: train_end_id][["predict_target_value", "predict_target_label"]]
    df_target_test = df.loc[test_start_id: test_end_id][["predict_target_value", "predict_target_label"]]

    df_data_train.to_csv(f"{base_path}/stock_prices.{ticker_symbol}.data_train.csv")
    df_data_test.to_csv(f"{base_path}/stock_prices.{ticker_symbol}.data_test.csv")
    df_target_train.to_csv(f"{base_path}/stock_prices.{ticker_symbol}.target_train.csv")
    df_target_test.to_csv(f"{base_path}/stock_prices.{ticker_symbol}.target_test.csv")


if __name__ == "__main__":
    execute()
