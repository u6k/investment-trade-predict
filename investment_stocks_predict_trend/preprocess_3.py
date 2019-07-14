import joblib
import pandas as pd

from app_logging import get_app_logger
import app_s3


def execute():
    L = get_app_logger()
    L.info("start")

    s3_bucket = "u6k"
    input_base_path_preprocess = "ml-data/stocks/preprocess_2.test"
    input_base_path_simulate = "ml-data/stocks/simulate_trade_2.test"
    output_base_path = "ml-data/stocks/preprocess_3.test"

    train_start_date = "2008-01-01"
    train_end_date = "2017-12-31"
    test_start_date = "2018-01-01"
    test_end_date = "2018-12-31"

    df_companies = app_s3.read_dataframe(s3_bucket, f"{input_base_path_preprocess}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(preprocess)(ticker_symbol, s3_bucket, input_base_path_preprocess, input_base_path_simulate, output_base_path) for ticker_symbol in df_companies.index])
    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(train_test_split)(ticker_symbol, s3_bucket, output_base_path, train_start_date, train_end_date, test_start_date, test_end_date) for ticker_symbol in df_companies.index])

    for result in results:
        ticker_symbol = result["ticker_symbol"]

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]
        df_companies_result.at[ticker_symbol, "message"] = result["message"]

    app_s3.write_dataframe(df_companies_result, s3_bucket, f"{output_base_path}/companies.csv")
    df_companies_result.to_csv("local/companies.preprocess_3.csv")
    L.info("finish")


def preprocess(ticker_symbol, s3_bucket, input_base_path_preprocess, input_base_path_simulate, output_base_path):
    L = get_app_logger(f"preprocess.{ticker_symbol}")
    L.info(f"preprocess: {ticker_symbol}")

    try:
        df_preprocess = app_s3.read_dataframe(s3_bucket, f"{input_base_path_preprocess}/stock_prices.{ticker_symbol}.csv", index_col=0)
        df_simulate = app_s3.read_dataframe(s3_bucket, f"{input_base_path_simulate}/stock_prices.{ticker_symbol}.csv", index_col=0)

        df = df_preprocess[[
            "date",
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
        ]].copy()

        df["predict_target_value"] = df_simulate["profit_rate"].shift(-1)
        df["predict_target_label"] = df["predict_target_value"].apply(lambda v: 1 if v > 1.0 else 0)

        app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")

        message = ""
    except Exception as err:
        L.exception(err)
        message = err.__str__()

    return {
        "ticker_symbol": ticker_symbol,
        "message": message
    }


def train_test_split(ticker_symbol, s3_bucket, base_path, train_start_date, train_end_date, test_start_date, test_end_date):
    L = get_app_logger(f"train_test_split.{ticker_symbol}")
    L.info(f"train_test_split: {ticker_symbol}")

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

        message = ""
    except Exception as err:
        L.exception(err)
        message = err.__str__()

    return {
        "ticker_symbol": ticker_symbol,
        "message": message
    }


if __name__ == "__main__":
    execute()
