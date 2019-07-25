import argparse

from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase


class SimulateTrade2(SimulateTradeBase):
    def simulate_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(f"{self._job_name}.simulate_impl.{ticker_symbol}")
        L.info(f"{self._job_name}.simulate_impl: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        losscut_rate = 0.95

        try:
            df = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            # simulate
            for start_id in df.index:
                # reset
                losscut_price = df.at[start_id, "open_price"] * losscut_rate
                end_id = None
                current_id = start_id

                while df.index[-1] > current_id:
                    # losscut
                    if df.at[current_id, "low_price"] < losscut_price:
                        end_id = current_id
                        break

                    # update losscut price
                    if losscut_price < (df.at[current_id, "open_price"] * losscut_rate):
                        losscut_price = df.at[current_id, "open_price"] * losscut_rate

                    current_id += 1

                # set result
                if end_id is not None:
                    df.at[start_id, "trade_end_id"] = end_id
                    df.at[start_id, "buy_price"] = df.at[start_id, "open_price"]
                    df.at[start_id, "sell_price"] = df.at[end_id, "low_price"]
                    df.at[start_id, "profit"] = df.at[end_id, "low_price"] - df.at[start_id, "open_price"]
                    df.at[start_id, "profit_rate"] = df.at[start_id, "profit"] / df.at[start_id, "sell_price"]

            app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def test_singles_impl(self, ticker_symbol, start_date, end_date, s3_bucket, input_preprocess_base_path, input_model_base_path, output_base_path):
        L = get_app_logger(f"test_singles_impl.{ticker_symbol}")
        L.info(f"test_singles_2: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        try:
            # Load data
            clf = app_s3.read_sklearn_model(s3_bucket, f"{input_model_base_path}/model.{ticker_symbol}.joblib")
            df = app_s3.read_dataframe(s3_bucket, f"{input_preprocess_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            df_prices = df[["date", "open_price", "high_price", "low_price", "close_price", "adjusted_close_price", "volume"]].copy()
            df_preprocessed = df.drop(["date", "open_price", "high_price", "low_price", "close_price", "adjusted_close_price", "volume", "predict_target"], axis=1)

            # Predict
            target_period_ids = df_prices.query(f"'{start_date}' <= date <= '{end_date}'").index
            df_prices = df_prices.loc[target_period_ids[0]-1: target_period_ids[-1]+1]
            data = df_preprocessed.loc[target_period_ids[0]-1: target_period_ids[-1]+1].values
            df_prices = df_prices.assign(predict=clf.predict(data))

            # Backtest
            losscut_rate = 0.95

            buy_price = None
            losscut_price = None

            for id in target_period_ids:
                # Buy
                if buy_price is None and df_prices.at[id-1, "predict"] == 1:
                    buy_price = df_prices.at[id, "open_price"]
                    losscut_price = buy_price * losscut_rate

                    df_prices.at[id, "action"] = "buy"

                # Sell
                if losscut_price is not None and df_prices.at[id, "low_price"] < losscut_price:
                    sell_price = df_prices.at[id, "low_price"]
                    profit = sell_price - buy_price
                    profit_rate = profit / sell_price

                    df_prices.at[id, "action"] = "sell"
                    df_prices.at[id, "buy_price"] = buy_price
                    df_prices.at[id, "sell_price"] = sell_price
                    df_prices.at[id, "profit"] = profit
                    df_prices.at[id, "profit_rate"] = profit_rate

                    buy_price = None
                    losscut_price = None

                # Turn end
                if losscut_price is not None and losscut_price < (df_prices.at[id, "open_price"] * losscut_rate):
                    losscut_price = df_prices.at[id, "open_price"] * losscut_rate

            app_s3.write_dataframe(df_prices, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")

        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="simulate, or test")
    parser.add_argument("--suffix", help="folder name suffix (default: test)", default="test")
    args = parser.parse_args()

    simulator = SimulateTrade2("simulate_trade_2")

    if args.task == "simulate":
        simulator.simulate(
            s3_bucket="u6k",
            input_base_path=f"ml-data/stocks/preprocess_1.{args.suffix}",
            output_base_path=f"ml-data/stocks/simulate_trade_2.{args.suffix}"
        )
        simulator.simulate_report(
            start_date="2018-01-01",
            end_date="2018-12-31",
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/simulate_trade_2.{args.suffix}"
        )
    elif args.task == "test":
        simulator.test_singles(
            start_date="2018-01-01",
            end_date="2018-12-31",
            s3_bucket="u6k",
            input_preprocess_base_path=f"ml-data/stocks/predict_3.simulate_trade_2.{args.suffix}",
            input_model_base_path=f"ml-data/stocks/predict_3.simulate_trade_2.{args.suffix}",
            output_base_path=f"ml-data/stocks/simulate_trade_2_test.{args.suffix}"
        )

        simulator.report_singles(
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/simulate_trade_2_test.{args.suffix}"
        )
    else:
        parser.print_help()
