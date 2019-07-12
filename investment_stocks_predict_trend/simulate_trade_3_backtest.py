import joblib
import pandas as pd

from app_logging import get_app_logger


def backtest_singles():
    L = get_app_logger()
    L.info("start")

    input_base_path = "local/backtest_preprocessed"
    output_base_path = "local/simulate_trade_3_backtest"

    start_date = "2018-01-01"
    end_date = "2018-12-31"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(backtest_singles_impl)(ticker_symbol, input_base_path, output_base_path, start_date, end_date) for ticker_symbol in df_companies.index])

    for result in results:
        ticker_symbol = result["ticker_symbol"]

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]
        df_companies_result.at[ticker_symbol, "message"] = result["message"]
        df_companies_result.at[ticker_symbol, "asset"] = result["asset"]
        df_companies_result.at[ticker_symbol, "trade_count"] = result["trade_count"]

    df_companies_result.to_csv(f"{output_base_path}/companies.csv")
    L.info("finish")


def backtest_singles_impl(ticker_symbol, input_base_path, output_base_path, start_date, end_date):
    L = get_app_logger(f"backtest_singles.{ticker_symbol}")
    L.info(f"backtest_singles: {ticker_symbol}")

    try:
        # Backtest
        asset = 0
        trade_count = 0

        # Load data
        clf = joblib.load(f"{input_base_path}/model.{ticker_symbol}.joblib")
        df_prices = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
        df_preprocessed = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.preprocessed.csv", index_col=0) \
            .drop(["date", "predict_target_value", "predict_target_label"], axis=1)

        # Predict
        target_period_ids = df_prices.query(f"'{start_date}' <= date <= '{end_date}'").index
        df_prices = df_prices.loc[target_period_ids[0]-1: target_period_ids[-1]]
        data = df_preprocessed.loc[target_period_ids[0]-1: target_period_ids[-1]].values
        df_prices = df_prices.assign(predict=clf.predict(data))

        for id in target_period_ids:
            # Trade
            if df_prices.at[id-1, "predict"] == 1:
                profit = df_prices.at[id, "close_price"] - df_prices.at[id, "open_price"]
                asset += profit
                trade_count += 1

                df_prices.at[id, "action"] = "trade"
                df_prices.at[id, "profit"] = profit
                df_prices.at[id, "profit_rate"] = profit/df_prices.at[id, "open_price"]

            # Turn end
            df_prices.at[id, "asset"] = asset

        df_prices.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.csv")

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
