import argparse
from datetime import datetime
import pandas as pd

from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase


class SimulateTrade4(SimulateTradeBase):
    def simulate_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(f"{self._job_name}.simulate_impl.{ticker_symbol}")
        L.info(f"{self._job_name}.simulate_impl: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        compare_high_price_period = 10
        losscut_rate = 0.95
        take_profit_rate = 0.95

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

            past_high_price_columns.append("past_high_price_max")
            df = df.drop(past_high_price_columns, axis=1)

            # Calc profit
            for id in df.query("buy_signal==1").index:
                buy_id = id + 1
                buy_price = df.at[buy_id, "open_price"]
                losscut_price = buy_price * losscut_rate
                take_profit_price = buy_price * take_profit_rate
                sell_id = None
                sell_price = None
                take_profit = False

                for id in df.loc[buy_id:].index:
                    if take_profit:
                        sell_id = id
                        sell_price = df.at[id, "open_price"]
                        break

                    if df.at[id, "low_price"] < losscut_price:
                        sell_id = id
                        sell_price = df.at[id, "low_price"]
                        break

                    if df.at[id, "high_price"] < take_profit_price:
                        take_profit = True

                    losscut_price_tmp = df.at[id, "high_price"] * losscut_rate
                    if losscut_price_tmp > losscut_price:
                        losscut_price = losscut_price_tmp

                    take_profit_price_tmp = df.at[id, "high_price"] * take_profit_rate
                    if take_profit_price_tmp > take_profit_price:
                        take_profit_price = take_profit_price_tmp

                if sell_id is not None:
                    df.at[buy_id, "buy_price"] = buy_price
                    df.at[buy_id, "sell_id"] = sell_id
                    df.at[buy_id, "sell_price"] = sell_price
                    df.at[buy_id, "profit"] = sell_price - buy_price
                    df.at[buy_id, "profit_rate"] = (sell_price - buy_price) / sell_price

            # Save data
            app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def test_singles_impl(self, ticker_symbol, start_date, end_date, s3_bucket, input_preprocess_base_path, input_model_base_path, output_base_path):
        L = get_app_logger(f"test_singles_impl.{ticker_symbol}")
        L.info(f"test_singles_4: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        losscut_rate = 0.95
        take_profit_rate = 0.95

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

            # Test
            for id in df_prices.query("predict==1").index:
                buy_id = id + 1
                buy_price = df_prices.at[buy_id, "open_price"]
                losscut_price = buy_price * losscut_rate
                take_profit_price = buy_price * take_profit_rate
                sell_id = None
                sell_price = None
                take_profit = False

                for id in df_prices.loc[buy_id:].index:
                    if take_profit:
                        sell_id = id
                        sell_price = df_prices.at[id, "open_price"]
                        break

                    if df_prices.at[id, "low_price"] < losscut_price:
                        sell_id = id
                        sell_price = df_prices.at[id, "low_price"]
                        break

                    if df.at[id, "high_price"] < take_profit_price:
                        take_profit = True

                    losscut_price_tmp = df_prices.at[id, "high_price"] * losscut_rate
                    if losscut_price_tmp > losscut_price:
                        losscut_price = losscut_price_tmp

                    take_profit_price_tmp = df_prices.at[id, "high_price"] * take_profit_rate
                    if take_profit_price_tmp > take_profit_price:
                        take_profit_price = take_profit_price_tmp

                if sell_id is not None:
                    df_prices.at[buy_id, "buy_price"] = buy_price
                    df_prices.at[buy_id, "sell_id"] = sell_id
                    df_prices.at[buy_id, "sell_price"] = sell_price
                    df_prices.at[buy_id, "profit"] = sell_price - buy_price
                    df_prices.at[buy_id, "profit_rate"] = (sell_price - buy_price) / sell_price

            app_s3.write_dataframe(df_prices, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def test_all(self, start_date, end_date, s3_bucket, base_path):
        L = get_app_logger("test_all")
        L.info("start")

        df_action = pd.DataFrame(columns=["date", "ticker_symbol", "action", "price", "stocks", "profit", "profit_rate", "fee", "tax"])
        df_stocks = pd.DataFrame(columns=["buy_price", "buy_stocks", "losscut_price", "take_profit_price", "take_profit", "close_price_latest"])
        df_result = pd.DataFrame(columns=["fund", "asset"])

        df_report = app_s3.read_dataframe(s3_bucket, f"{base_path}/report.csv", index_col=0)

        df_prices_dict = {}
        for ticker_symbol in df_report.query("trade_count>10 and profit_factor>2.0").sort_values("expected_value", ascending=False).head(50).index:
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
        losscut_rate = 0.95
        take_profit_rate = 0.95

        for date in self.date_range(start_date, end_date):
            date_str = date.strftime("%Y-%m-%d")
            L.info(f"test_all: {date_str}")

            # Sell: take profit
            for ticker_symbol in df_stocks.index:
                if not df_stocks.at[ticker_symbol, "take_profit"]:
                    continue

                df_prices = df_prices_dict[ticker_symbol]

                if len(df_prices.query(f"date=='{date_str}'")) == 0:
                    continue

                prices_id = df_prices.query(f"date=='{date_str}'").index[0]

                sell_price = df_prices.at[prices_id, "open_price"]
                buy_price = df_stocks.at[ticker_symbol, "buy_price"]
                buy_stocks = df_stocks.at[ticker_symbol, "buy_stocks"]
                profit = (sell_price - buy_price) * buy_stocks
                profit_rate = (sell_price - buy_price) / sell_price
                fee = sell_price * buy_stocks * fee_rate
                tax = profit * tax_rate if profit > 0 else 0

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date_str
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = "sell"
                df_action.at[action_id, "price"] = sell_price
                df_action.at[action_id, "stocks"] = buy_stocks
                df_action.at[action_id, "profit"] = profit
                df_action.at[action_id, "profit_rate"] = profit_rate
                df_action.at[action_id, "fee"] = fee
                df_action.at[action_id, "tax"] = tax

                df_stocks = df_stocks.drop(ticker_symbol)

                fund += profit-fee-tax

            # Sell: losscut
            for ticker_symbol in df_stocks.index:
                df_prices = df_prices_dict[ticker_symbol]

                if len(df_prices.query(f"date=='{date_str}'")) == 0:
                    continue

                prices_id = df_prices.query(f"date=='{date_str}'").index[0]

                if df_prices.at[prices_id, "low_price"] >= df_stocks.at[ticker_symbol, "losscut_price"]:
                    continue

                sell_price = df_prices.at[prices_id, "open_price"]
                buy_price = df_stocks.at[ticker_symbol, "buy_price"]
                buy_stocks = df_stocks.at[ticker_symbol, "buy_stocks"]
                profit = (sell_price - buy_price) * buy_stocks
                profit_rate = (sell_price - buy_price) / sell_price
                fee = sell_price * buy_stocks * fee_rate
                tax = profit * tax_rate if profit > 0 else 0

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date_str
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = "sell"
                df_action.at[action_id, "price"] = sell_price
                df_action.at[action_id, "stocks"] = buy_stocks
                df_action.at[action_id, "profit"] = profit
                df_action.at[action_id, "profit_rate"] = profit_rate
                df_action.at[action_id, "fee"] = fee
                df_action.at[action_id, "tax"] = tax

                df_stocks = df_stocks.drop(ticker_symbol)

                fund += profit-fee-tax

            # Flag take profit
            for ticker_symbol in df_stocks.index:
                df_prices = df_prices_dict[ticker_symbol]

                if len(df_prices.query(f"date=='{date_str}'")) == 0:
                    continue

                prices_id = df_prices.query(f"date=='{date_str}'").index[0]

                if df_prices.at[prices_id, "high_price"] < df_stocks.at[ticker_symbol, "take_profit_price"]:
                    df_stocks.at[ticker_symbol, "take_profit"] = True

            # Buy
            for ticker_symbol in df_prices_dict.keys():
                if ticker_symbol in df_stocks.index:
                    continue

                df_prices = df_prices_dict[ticker_symbol]

                if len(df_prices.query(f"date=='{date_str}'")) == 0:
                    continue

                prices_id = df_prices.query(f"date=='{date_str}'").index[0]

                if df_prices.at[prices_id, "buy_price"] is None:
                    continue

                buy_price = df_prices.at[prices_id, "open_price"]
                buy_stocks = asset * available_rate // buy_price

                if buy_stocks <= 0:
                    continue

                if (fund - buy_price * buy_stocks) < (asset * total_available_rate):
                    continue

                fee = buy_price * buy_stocks * fee_rate
                losscut_price = buy_price * losscut_rate
                take_profit_price = buy_price * take_profit_rate

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date_str
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = "buy"
                df_action.at[action_id, "price"] = buy_price
                df_action.at[action_id, "stocks"] = buy_stocks
                df_action.at[action_id, "fee"] = fee

                df_stocks.at[ticker_symbol, "buy_price"] = buy_price
                df_stocks.at[ticker_symbol, "buy_stocks"] = buy_stocks
                df_stocks.at[ticker_symbol, "losscut_price"] = losscut_price
                df_stocks.at[ticker_symbol, "take_profit_price"] = take_profit_price
                df_stocks.at[ticker_symbol, "take_profit"] = False
                df_stocks.at[ticker_symbol, "close_price_latest"] = df_prices.at[prices_id, "close_price"]

                fund -= buy_price * buy_stocks + fee

            # Update losscut, take profit
            for ticker_symbol in df_stocks.index:
                df_prices = df_prices_dict[ticker_symbol]

                if len(df_prices.query(f"date=='{date_str}'")) == 0:
                    continue

                prices_id = df_prices.query(f"date=='{date_str}'").index[0]

                losscut_price_tmp = df_prices.at[prices_id, "high_price"] * losscut_rate
                if losscut_price_tmp > df_stocks.at[ticker_symbol, "losscut_price"]:
                    df_stocks.at[ticker_symbol, "losscut_price"] = losscut_price_tmp

                take_profit_price_tmp = df_prices.at[prices_id, "high_price"] * take_profit_rate
                if take_profit_price_tmp > df_stocks.at[ticker_symbol, "take_profit_price"]:
                    df_stocks.at[ticker_symbol, "take_profit_price"] = take_profit_price_tmp

            asset = fund
            for ticker_symbol in df_stocks.index:
                asset += df_stocks.at[ticker_symbol, "close_price_latest"] * df_stocks.at[ticker_symbol, "buy_stocks"]

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

    simulator = SimulateTrade4("simulate_trade_4")

    if args.task == "simulate":
        simulator.simulate(
            s3_bucket="u6k",
            input_base_path=f"ml-data/stocks/preprocess_1.{args.suffix}",
            output_base_path=f"ml-data/stocks/simulate_trade_4.{args.suffix}"
        )

        simulator.simulate_report(
            start_date="2018-01-01",
            end_date="2018-12-31",
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/simulate_trade_4.{args.suffix}"
        )
    elif args.task == "test":
        simulator.test_singles(
            start_date="2018-01-01",
            end_date="2018-12-31",
            s3_bucket="u6k",
            input_preprocess_base_path=f"ml-data/stocks/predict_3.simulate_trade_4.{args.suffix}",
            input_model_base_path=f"ml-data/stocks/predict_3.simulate_trade_4.{args.suffix}",
            output_base_path=f"ml-data/stocks/simulate_trade_4_test.{args.suffix}"
        )

        simulator.report_singles(
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/simulate_trade_4_test.{args.suffix}"
        )
    elif args.task == "test_all":
        simulator.test_all(
            start_date=datetime(2018, 1, 1),
            end_date=datetime(2019, 1, 1),
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/simulate_trade_4_test.{args.suffix}"
        )
    else:
        parser.print_help()
