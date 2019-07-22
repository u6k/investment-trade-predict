import argparse
from datetime import datetime
import pandas as pd

from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase


class SimulateTrade3(SimulateTradeBase):
    def simulate_singles_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(f"simulate_singles_impl.{ticker_symbol}")
        L.info(f"simulate_trade_3: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        try:
            df = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            df["profit"] = df["close_price"] - df["open_price"]
            df["profit_rate"] = df["profit"] / df["close_price"]

            app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def test_singles_impl(self, ticker_symbol, start_date, end_date, s3_bucket, input_preprocess_base_path, input_model_base_path, output_base_path):
        L = get_app_logger(f"test_singles_impl.{ticker_symbol}")
        L.info(f"test_singles_3: {ticker_symbol}")

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
            df_prices = df_prices.loc[target_period_ids[0]-1: target_period_ids[-1]]
            data = df_preprocessed.loc[target_period_ids[0]-1: target_period_ids[-1]].values
            df_prices = df_prices.assign(predict=clf.predict(data))

            # Backtest
            for id in target_period_ids:
                # Trade
                if df_prices.at[id-1, "predict"] == 1:
                    buy_price = df_prices.at[id, "open_price"]
                    sell_price = df_prices.at[id, "close_price"]
                    profit = sell_price - buy_price
                    profit_rate = profit / sell_price

                    df_prices.at[id, "action"] = "trade"
                    df_prices.at[id, "buy_price"] = buy_price
                    df_prices.at[id, "sell_price"] = sell_price
                    df_prices.at[id, "profit"] = profit
                    df_prices.at[id, "profit_rate"] = profit_rate

            app_s3.write_dataframe(df_prices, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def test_all(self, start_date, end_date, s3_bucket, base_path):
        L = get_app_logger("test_all")
        L.info("start")

        df_action = pd.DataFrame(columns=["date", "ticker_symbol", "action", "price", "stocks", "profit", "profit_rate"])
        df_result = pd.DataFrame(columns=["fund", "asset"])

        df_report = app_s3.read_dataframe(s3_bucket, f"{base_path}/report.csv", index_col=0)

        df_prices_dict = {}
        for ticker_symbol in df_report.query("trade_count>50 and profit_factor>2.0").sort_values("expected_value", ascending=False).index:
            L.info(f"load data: {ticker_symbol}")
            df_prices_dict[ticker_symbol] = app_s3.read_dataframe(s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        fund = 100000
        asset = fund
        available_rate = 0.05
        total_available_rate = 0.5
        fee_rate = 0.001
        tax_rate = 0.21

        for date in self.date_range(start_date, end_date):
            date_str = date.strftime("%Y-%m-%d")
            L.info(f"test_all: {date_str}")

            # Trade
            for ticker_symbol in df_prices_dict.keys():
                df_prices = df_prices_dict[ticker_symbol]

                if len(df_prices.query(f"date=='{date_str}'")) == 0:
                    continue

                prices_id = df_prices.query(f"date=='{date_str}'").index[0]

                if df_prices.at[prices_id, "action"] != "trade":
                    continue

                # Buy
                buy_price = df_prices.at[prices_id, "open_price"]
                buy_stocks = asset * available_rate // buy_price

                if buy_stocks <= 0:
                    continue

                if (fund - buy_price * buy_stocks) < (asset * total_available_rate):
                    continue

                fund -= buy_price * buy_stocks

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date_str
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = "buy"
                df_action.at[action_id, "price"] = buy_price
                df_action.at[action_id, "stocks"] = buy_stocks

                fee_price = (buy_price * buy_stocks) * fee_rate
                fund -= fee_price

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date_str
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = "fee"
                df_action.at[action_id, "price"] = fee_price
                df_action.at[action_id, "stocks"] = 1
                df_action.at[action_id, "profit"] = -1 * fee_price

                # Sell
                sell_price = df_prices.at[prices_id, "close_price"]

                profit = (sell_price - buy_price) * buy_stocks
                profit_rate = profit / (sell_price * buy_stocks)

                fund += sell_price * buy_stocks

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date_str
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = "sell"
                df_action.at[action_id, "price"] = sell_price
                df_action.at[action_id, "stocks"] = buy_stocks
                df_action.at[action_id, "profit"] = profit
                df_action.at[action_id, "profit_rate"] = profit_rate

                fee_price = (sell_price * buy_stocks) * fee_rate
                fund -= fee_price

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date_str
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = "fee"
                df_action.at[action_id, "price"] = fee_price
                df_action.at[action_id, "stocks"] = 1
                df_action.at[action_id, "profit"] = -1 * fee_price

                if profit > 0:
                    tax_price = profit * tax_rate
                    fund -= tax_price

                    action_id = len(df_action)
                    df_action.at[action_id, "date"] = date_str
                    df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                    df_action.at[action_id, "action"] = "tax"
                    df_action.at[action_id, "price"] = tax_price
                    df_action.at[action_id, "stocks"] = 1
                    df_action.at[action_id, "profit"] = -1 * tax_price

            # Turn end
            asset = fund

            df_result.at[date_str, "fund"] = fund
            df_result.at[date_str, "asset"] = asset

            L.info(df_result.loc[date_str])

        app_s3.write_dataframe(df_action, s3_bucket, f"{base_path}/test_all.action.csv")
        app_s3.write_dataframe(df_result, s3_bucket, f"{base_path}/test_all.result.csv")

        L.info("finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="simulate, test, or test_all")
    parser.add_argument("--suffix", help="folder name suffix (default: test)", default="test")
    args = parser.parse_args()

    if args.task == "simulate":
        SimulateTrade3().simulate_singles(
            s3_bucket="u6k",
            input_base_path=f"ml-data/stocks/preprocess_1.{args.suffix}",
            output_base_path=f"ml-data/stocks/simulate_trade_3.{args.suffix}"
        )
    elif args.task == "test":
        SimulateTrade3().test_singles(
            start_date="2018-01-01",
            end_date="2018-12-31",
            s3_bucket="u6k",
            input_preprocess_base_path=f"ml-data/stocks/predict_3.simulate_trade_3.{args.suffix}",
            input_model_base_path=f"ml-data/stocks/predict_3.simulate_trade_3.{args.suffix}",
            output_base_path=f"ml-data/stocks/simulate_trade_3_test.{args.suffix}"
        )

        SimulateTrade3().report_singles(
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/simulate_trade_3_test.{args.suffix}"
        )
    elif args.task == "test_all":
        SimulateTrade3().test_all(
            start_date=datetime(2018, 1, 1),
            end_date=datetime(2019, 1, 1),
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/simulate_trade_3_test.{args.suffix}"
        )
    else:
        parser.print_help()
