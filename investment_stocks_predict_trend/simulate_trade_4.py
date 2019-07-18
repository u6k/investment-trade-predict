from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase


class SimulateTrade4(SimulateTradeBase):
    def simulate_singles_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(ticker_symbol)
        L.info(f"simulate_trade_4: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        compare_high_price_period = 5
        hold_period = 5

        try:
            # Load data
            df = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            # Setting buy signal
            past_high_price_columns = []
            for i in range(1, compare_high_price_period+1):
                df[f"past_high_price_{i}"] = df["high_price"].shift(i)
                past_high_price_columns.append(f"past_high_price_{i}")

            df["past_high_price_max"] = df[past_high_price_columns].max(axis=1)
            for id in df.index:
                df.at[id, "buy_signal"] = 1 if df.at[id, "high_price"] > df.at[id, "past_high_price_max"] else 0

            # Calc profit
            df["buy_price"] = df["open_price"].shift(-1)
            df["sell_price"] = df["open_price"].shift(-hold_period-1)
            df["profit"] = df["sell_price"] - df["buy_price"]
            df["profit_rate"] = df["profit"] / df["sell_price"]

            # Drop dust data
            for id in df.index:
                if df.at[id, "buy_signal"] == 0:
                    df.at[id, "profit"] = None
                    df.at[id, "profit_rate"] = None

            df = df.drop(past_high_price_columns, axis=1)
            df = df.drop(["past_high_price_max", "buy_signal", "buy_price", "sell_price"], axis=1)

            # Save data
            app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def backtest_singles_impl(self, ticker_symbol, start_date, end_date, s3_bucket, input_prices_base_path, input_preprocess_base_path, input_model_base_path, output_base_path):
        L = get_app_logger(f"backtest_singles.{ticker_symbol}")
        L.info(f"backtest_singles_4: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        hold_period = 5

        try:
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

            # Backtest
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

                    df_prices.at[id, "action"] = "sell"
                    df_prices.at[id, "buy_price"] = buy_price
                    df_prices.at[id, "sell_price"] = sell_price
                    df_prices.at[id, "profit"] = profit
                    df_prices.at[id, "profit_rate"] = profit_rate

                    buy_price = None
                    hold_days_remain = None

                # Turn end
                if hold_days_remain is not None:
                    hold_days_remain -= 1

            app_s3.write_dataframe(df_prices, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result


if __name__ == "__main__":
    s3_bucket = "u6k"
    input_base_path = "ml-data/stocks/preprocess_1.test"
    output_base_path = "ml-data/stocks/simulate_trade_4.test"

    SimulateTrade4().simulate_singles(
        s3_bucket,
        input_base_path,
        output_base_path
    )
