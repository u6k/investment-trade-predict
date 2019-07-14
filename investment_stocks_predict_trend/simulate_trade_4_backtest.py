import joblib
import pandas as pd

from app_logging import get_app_logger
import app_s3


def backtest_singles():
    L = get_app_logger()
    L.info("start")

    s3_bucket = "u6k"
    input_prices_base_path = "ml-data/stocks/preprocess_1.test"
    input_preprocess_base_path = "ml-data/stocks/preprocess_5.test"
    input_model_base_path = "ml-data/stocks/predict_5_preprocess_5.test"
    output_base_path = "ml-data/stocks/simulate_trade_4_backtest.test"

    start_date = "2018-01-01"
    end_date = "2018-12-31"

    df_companies = app_s3.read_dataframe(s3_bucket, f"{input_prices_base_path}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(backtest_singles_impl)(ticker_symbol, s3_bucket, input_prices_base_path, input_preprocess_base_path, input_model_base_path, output_base_path, start_date, end_date) for ticker_symbol in df_companies.index])

    for result in results:
        ticker_symbol = result["ticker_symbol"]

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]
        df_companies_result.at[ticker_symbol, "message"] = result["message"]
        df_companies_result.at[ticker_symbol, "asset"] = result["asset"]
        df_companies_result.at[ticker_symbol, "trade_count"] = result["trade_count"]

    app_s3.write_dataframe(df_companies_result, s3_bucket, f"{output_base_path}/companies.csv")
    df_companies_result.to_csv("companies.simulate_trade_4_backtest.csv")
    L.info("finish")


def backtest_singles_impl(ticker_symbol, s3_bucket, input_prices_base_path, input_preprocess_base_path, input_model_base_path, output_base_path, start_date, end_date):
    L = get_app_logger(f"backtest_singles.{ticker_symbol}")
    L.info(f"backtest_singles: {ticker_symbol}")

    hold_period = 5

    try:
        # Backtest
        asset = 0
        trade_count = 0
        buy_price = None
        hold_days_remain = None

        # Load data
        clf = app_s3.read_sklearn_model(s3_bucket, f"{input_model_base_path}/model.{ticker_symbol}.joblib")
        df_prices = app_s3.read_dataframe(s3_bucket, f"{input_prices_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
        df_preprocessed = app_s3.read_dataframe(s3_bucket, f"{input_preprocess_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0) \
            .drop(["date", "predict_target_value", "predict_target_label"], axis=1)

        # Predict
        target_period_ids = df_prices.query(f"'{start_date}' <= date <= '{end_date}'").index
        df_prices = df_prices.loc[target_period_ids[0]-1: target_period_ids[-1]]
        data = df_preprocessed.loc[target_period_ids[0]-1: target_period_ids[-1]].values
        df_prices = df_prices.assign(predict=clf.predict(data))

        for id in target_period_ids:
            # Buy
            if buy_price is None and df_prices.at[id-1, "predict"] == 1:
                buy_price = df_prices.at[id, "open_price"]
                hold_days_remain = hold_period

                df_prices.at[id, "action"] = "buy"

            # Sell
            if hold_days_remain == 0:
                sell_price = df_prices.at[id, "open_price"]
                profit = sell_price - buy_price
                profit_rate = profit / sell_price
                asset += profit
                trade_count += 1

                df_prices.at[id, "action"] = "sell"
                df_prices.at[id, "profit"] = profit
                df_prices.at[id, "profit_rate"] = profit_rate

                buy_price = None
                hold_days_remain = None

            # Turn end
            if hold_days_remain is not None:
                hold_days_remain -= 1

            df_prices.at[id, "asset"] = asset

        app_s3.write_dataframe(df_prices, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")

        message = ""
    except Exception as err:
        L.exception(err)
        message = err.__str__()

    L.info(f"ticker_symbol={ticker_symbol}, message={message}, asset={asset}, trade_count={trade_count}")

    return {
        "ticker_symbol": ticker_symbol,
        "message": message,
        "asset": asset,
        "trade_count": trade_count
    }


if __name__ == "__main__":
    backtest_singles()
