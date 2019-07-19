import joblib
import pandas as pd

from app_logging import get_app_logger
import app_s3


def execute():
    L = get_app_logger()
    L.info("start")

    s3_bucket = "u6k"
    input_base_path_preprocess = "ml-data/stocks/preprocess_2.test"
    input_base_path_simulate = "ml-data/stocks/simulate_trade_6.test"
    output_base_path = "ml-data/stocks/preprocess_7.test"

    train_start_date = "2008-01-01"
    train_end_date = "2017-12-31"
    test_start_date = "2018-01-01"
    test_end_date = "2018-12-31"

    df_companies = app_s3.read_dataframe(s3_bucket, f"{input_base_path_preprocess}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(preprocess)(ticker_symbol, s3_bucket, input_base_path_preprocess, input_base_path_simulate, output_base_path) for ticker_symbol in df_companies.index])
    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(train_test_split)(ticker_symbol, s3_bucket, output_base_path, train_start_date, train_end_date, test_start_date, test_end_date) for ticker_symbol in df_companies.index])

    for result in results:
        if result["exception"] is not None:
            continue

        ticker_symbol = result["ticker_symbol"]
        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

    app_s3.write_dataframe(df_companies_result, s3_bucket, f"{output_base_path}/companies.csv")

    L.info("finish")


def preprocess(ticker_symbol, s3_bucket, input_base_path_preprocess, input_base_path_simulate, output_base_path):
    L = get_app_logger(f"preprocess.{ticker_symbol}")
    L.info(f"preprocess_7: {ticker_symbol}")

    result = {
        "ticker_symbol": ticker_symbol,
        "exception": None
    }

    try:
        df_preprocess = app_s3.read_dataframe(s3_bucket, f"{input_base_path_preprocess}/stock_prices.{ticker_symbol}.csv", index_col=0)
        df_simulate = app_s3.read_dataframe(s3_bucket, f"{input_base_path_simulate}/stock_prices.{ticker_symbol}.csv", index_col=0)

        df = df_preprocess[[
            "date",
            "volume_change_minmax",
            "adjusted_close_price_change_minmax",
            "sma_5_minmax",
            "sma_10_minmax",
            "sma_20_minmax",
            "sma_40_minmax",
            "sma_80_minmax",
            "momentum_5_minmax",
            "momentum_10_minmax",
            "momentum_20_minmax",
            "momentum_40_minmax",
            "momentum_80_minmax",
            "roc_5_minmax",
            "roc_10_minmax",
            "roc_20_minmax",
            "roc_40_minmax",
            "roc_80_minmax",
            "rsi_5_minmax",
            "rsi_10_minmax",
            "rsi_14_minmax",
            "rsi_20_minmax",
            "rsi_40_minmax",
            "stochastic_k_5_minmax",
            "stochastic_d_5_minmax",
            "stochastic_sd_5_minmax",
            "stochastic_k_9_minmax",
            "stochastic_d_9_minmax",
            "stochastic_sd_9_minmax",
            "stochastic_k_20_minmax",
            "stochastic_d_20_minmax",
            "stochastic_sd_20_minmax",
            "stochastic_k_25_minmax",
            "stochastic_d_25_minmax",
            "stochastic_sd_25_minmax",
            "stochastic_k_40_minmax",
            "stochastic_d_40_minmax",
            "stochastic_sd_40_minmax"
        ]].copy()

        df["predict_target_value"] = df_simulate["profit_rate"]
        df["predict_target_label"] = df["predict_target_value"].apply(lambda v: 1 if v > 0.0 else 0)

        app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
    except Exception as err:
        L.exception(f"ticker_symbol={ticker_symbol}, {err}")
        result["exception"] = err

    return result


def train_test_split(ticker_symbol, s3_bucket, base_path, train_start_date, train_end_date, test_start_date, test_end_date):
    L = get_app_logger(f"train_test_split.{ticker_symbol}")
    L.info(f"train_test_split_6: {ticker_symbol}")

    result = {
        "ticker_symbol": ticker_symbol,
        "exception": None
    }

    try:
        df = app_s3.read_dataframe(s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0) \
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

        app_s3.write_dataframe(df_data_train, s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.data_train.csv")
        app_s3.write_dataframe(df_data_test, s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.data_test.csv")
        app_s3.write_dataframe(df_target_train, s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.target_train.csv")
        app_s3.write_dataframe(df_target_test, s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.target_test.csv")
    except Exception as err:
        L.exception(f"ticker_symbol={ticker_symbol}, {err}")
        result["exception"] = err

    return result


if __name__ == "__main__":
    execute()
