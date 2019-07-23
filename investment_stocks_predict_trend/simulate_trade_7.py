import argparse
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase


class SimulateTrade7(SimulateTradeBase):
    def test_singles_impl(self, ticker_symbol, start_date, end_date, s3_bucket, input_preprocess_base_path, input_model_base_path, output_base_path):
        L = get_app_logger(f"test_singles_impl.{ticker_symbol}")
        L.info(f"test_singles_3: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        try:
            # Load data
            model = app_s3.read_keras_model(s3_bucket, f"{input_model_base_path}/model.{ticker_symbol}")
            df = app_s3.read_dataframe(s3_bucket, f"{input_preprocess_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            df_prices = df[["date", "open_price", "high_price", "low_price", "close_price", "adjusted_close_price", "volume"]].copy()
            df_preprocessed = df.drop(["date", "open_price", "high_price", "low_price", "close_price", "adjusted_close_price", "volume", "predict_target"], axis=1)

            # Predict
            target_period_ids = df_prices.query(f"'{start_date}' <= date <= '{end_date}'").index
            df_prices = df_prices.loc[target_period_ids[0]-1: target_period_ids[-1]]
            data = df_preprocessed.loc[target_period_ids[0]-1: target_period_ids[-1]].values
            data = data.reshape(len(data), len(data[0]), 1)
            df_prices = df_prices.assign(predict=model.predict(data, batch_size=100, verbose=0))

            df_prices["profit_correct"] = df_prices["close_price"]-df_prices["open_price"]
            scaler = MinMaxScaler()
            df_prices["profit_correct_minmax"] = scaler.fit_transform(df_prices["profit_correct"].values.reshape(-1, 1))
            df_prices["profit_predict"] = scaler.inverse_transform(df_prices["predict"].values.reshape(-1, 1))

            # Backtest
            for id in target_period_ids:
                # Trade
                if df_prices.at[id-1, "profit_predict"] > 0:
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
        for ticker_symbol in df_report.query("trade_count>50").sort_values("profit_factor", ascending=False).head(50).index:
            if ticker_symbol in ["ni225", "topix", "djia"]:
                continue

            L.info(f"load data: {ticker_symbol}")
            df_prices_dict[ticker_symbol] = app_s3.read_dataframe(s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        fund = 1000000
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

                fee_price = (buy_price * buy_stocks) * fee_rate
                fund -= buy_price * buy_stocks + fee_price

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date_str
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = "buy"
                df_action.at[action_id, "price"] = buy_price
                df_action.at[action_id, "stocks"] = buy_stocks
                df_action.at[action_id, "fee"] = fee_price

                # Sell
                sell_price = df_prices.at[prices_id, "close_price"]

                profit = (sell_price - buy_price) * buy_stocks
                profit_rate = profit / (sell_price * buy_stocks)

                fee_price = (sell_price * buy_stocks) * fee_rate
                if profit > 0:
                    tax_price = profit * tax_rate
                else:
                    tax_price = 0
                fund += sell_price * buy_stocks - fee_price - tax_price

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date_str
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = "sell"
                df_action.at[action_id, "price"] = sell_price
                df_action.at[action_id, "stocks"] = buy_stocks
                df_action.at[action_id, "profit"] = profit
                df_action.at[action_id, "profit_rate"] = profit_rate
                df_action.at[action_id, "fee"] = fee_price
                df_action.at[action_id, "tax"] = tax_price

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
    parser.add_argument("--task", help="test, or test_all")
    parser.add_argument("--suffix", help="folder name suffix (default: test)", default="test")
    args = parser.parse_args()

    if args.task == "test":
        SimulateTrade7().test_singles(
            start_date="2018-01-01",
            end_date="2018-12-31",
            s3_bucket="u6k",
            input_preprocess_base_path=f"ml-data/stocks/predict_6.{args.suffix}",
            input_model_base_path=f"ml-data/stocks/predict_6.{args.suffix}",
            output_base_path=f"ml-data/stocks/simulate_trade_7_test.{args.suffix}"
        )

        SimulateTrade7().report_singles(
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/simulate_trade_7_test.{args.suffix}"
        )
    elif args.task == "test_all":
        SimulateTrade7().test_all(
            start_date=datetime(2018, 1, 1),
            end_date=datetime(2019, 1, 1),
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/simulate_trade_7_test.{args.suffix}"
        )
    else:
        parser.print_help()
