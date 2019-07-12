import joblib
import pandas as pd

from app_logging import get_app_logger


def backtest_singles():
    L = get_app_logger()
    L.info("start")

    input_base_path = "local/backtest_preprocessed"
    output_base_path = "local/simulate_trade_4_backtest"

    start_date = "2018-01-01"
    end_date = "2018-12-31"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(backtest_singles_impl)(ticker_symbol, input_base_path, output_base_path, start_date, end_date) for ticker_symbol in df_companies.index])

    for result in results:
        ticker_symbol = result[0]

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]
        df_companies_result.at[ticker_symbol, "message"] = result[1]
        df_companies_result.at[ticker_symbol, "asset"] = result[2]
        df_companies_result.at[ticker_symbol, "trade_count"] = result[3]

    df_companies_result.to_csv(f"{output_base_path}/companies.csv")
    L.info("finish")


def backtest_singles_impl(ticker_symbol, input_base_path, output_base_path, start_date, end_date):
    L = get_app_logger(f"backtest_single_impl.{ticker_symbol}")
    L.info(f"backtest_single: {ticker_symbol}")

    hold_period = 5

    try:
        # Backtest
        asset = 0
        trade_count = 0
        buy_price = None
        hold_days_remain = None

        # Load data
        clf = joblib.load(f"{input_base_path}/model.{ticker_symbol}.joblib")
        df_prices = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
        df_preprocessed = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.preprocessed.csv", index_col=0).drop(["date", "predict_target_value", "predict_target_label"], axis=1)

        id_array = df_prices.query(f"'{start_date}' <= date <= '{end_date}'").index

        for id in id_array:
            # Buy
            data = df_preprocessed.loc[id-1].values
            data = data.reshape(1, len(data))
            buy_signal = clf.predict(data)[0]

            if buy_price is None and buy_signal == 1:
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

        df_prices.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.csv")

        message = ""
    except Exception as err:
        L.exception(err)
        message = err.__str__()

    L.info(f"ticker_symbol={ticker_symbol}, message={message}, asset={asset}, trade_count={trade_count}")

    return (ticker_symbol, message, asset, trade_count)


if __name__ == "__main__":
    backtest_singles()
