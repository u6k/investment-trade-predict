import argparse
import pandas as pd

from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase
from predict_3 import PredictClassification_3


class SimulateTrade4(SimulateTradeBase):
    def simulate_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(f"{self._job_name}.simulate_impl.{ticker_symbol}")
        L.info(f"{self._job_name}.simulate_impl: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        compare_high_price_period = 5
        losscut_rate = 0.98
        take_profit_rate = 0.95
        minimum_profit_rate = 0.03

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

            # Simulate
            for id in df.query("buy_signal==1").index:
                buy_id = id + 1
                buy_price = df.at[buy_id, "open_price"]
                losscut_price = buy_price * losscut_rate
                take_profit_price = buy_price * take_profit_rate
                sell_id = None
                sell_price = None
                take_profit = False

                for id in df.loc[buy_id:].index:
                    # Sell: take profit
                    if take_profit:
                        sell_id = id
                        sell_price = df.at[id, "open_price"]
                        break

                    # Sell: losscut
                    if df.at[id, "low_price"] < losscut_price:
                        sell_id = id
                        sell_price = df.at[id, "low_price"]
                        break

                    # Flag: take profit
                    if df.at[id, "high_price"] < take_profit_price:
                        take_profit = True

                    # Update losscut/take profit price
                    losscut_price_tmp = df.at[id, "close_price"] * losscut_rate
                    if losscut_price_tmp > losscut_price:
                        losscut_price = losscut_price_tmp

                    take_profit_price_tmp = df.at[id, "high_price"] * take_profit_rate
                    if take_profit_price_tmp > take_profit_price:
                        take_profit_price = take_profit_price_tmp

                # Set result
                if sell_id is not None:
                    df.at[buy_id, "buy_date"] = df.at[buy_id, "date"]
                    df.at[buy_id, "buy_price"] = buy_price
                    df.at[buy_id, "sell_id"] = df.at[sell_id, "date"]
                    df.at[buy_id, "sell_price"] = sell_price
                    df.at[buy_id, "profit"] = sell_price - buy_price
                    df.at[buy_id, "profit_rate"] = (sell_price - buy_price) / sell_price

            # Labeling for predict
            df["predict_target_value"] = df["profit_rate"].shift(-1)
            df["predict_target_label"] = df["predict_target_value"].apply(lambda r: 1 if r >= minimum_profit_rate else 0)

            # Save data
            app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def forward_test_impl(self, ticker_symbol, predictor, s3_bucket, input_preprocess_base_path, input_simulate_base_path, output_base_path):
        L = get_app_logger(f"{self._job_name}.forward_test_impl.{ticker_symbol}")
        L.info(f"{self._job_name}.forward_test_impl: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        try:
            # Load data
            df_preprocess = app_s3.read_dataframe(s3_bucket, f"{input_preprocess_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
            df = app_s3.read_dataframe(s3_bucket, f"{input_simulate_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            # Predict
            df_data = df_preprocess.drop("date", axis=1).dropna()
            predict = predictor.model_predict(ticker_symbol, df_data)

            for i, id in enumerate(df_data.index):
                df.at[id, "predict"] = predict[i]

            # Test
            df["action"] = None

            for id in df.index[1:]:
                if df.at[id-1, "predict"] == 1 and df.at[id-1, "buy_signal"] == 1:
                    df.at[id, "action"] = "trade"
                else:
                    df.at[id, "sell_id"] = None
                    df.at[id, "buy_price"] = None
                    df.at[id, "sell_price"] = None
                    df.at[id, "profit"] = None
                    df.at[id, "profit_rate"] = None

            # Save data
            app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def forward_test_all(self, report_start_date, report_end_date, test_start_date, test_end_date, s3_bucket, base_path):
        L = get_app_logger(f"{self._job_name}.forward_test_all")
        L.info(f"{self._job_name}.forward_test_all: start")

        # Load data
        df_report = app_s3.read_dataframe(s3_bucket, f"{base_path}/report.{report_start_date}_{report_end_date}.csv", index_col=0)

        df_prices_dict = {}
        for ticker_symbol in df_report.query("profit_factor>2.0").sort_values("expected_value", ascending=False).head(100).index:
            L.info(f"load data: {ticker_symbol}")
            df_prices_dict[ticker_symbol] = app_s3.read_dataframe(s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        if len(df_prices_dict) == 0:
            for ticker_symbol in df_report.sort_values("expected_value", ascending=False).head(100).index:
                L.info(f"load data: {ticker_symbol}")
                df_prices_dict[ticker_symbol] = app_s3.read_dataframe(s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        # Init
        df_action = pd.DataFrame(columns=["date", "ticker_symbol", "action", "price", "stocks", "profit", "profit_rate", "fee", "tax"])
        df_stocks = pd.DataFrame(columns=["ticker_symbol", "buy_price", "buy_stocks", "close_price_latest", "sell_id", "sell_price"])
        df_result = pd.DataFrame(columns=["fund", "asset"])

        init_asset = 1000000
        fund = init_asset
        asset = init_asset
        available_rate = 0.05
        fee_rate = 0.001
        tax_rate = 0.21

        for date in self.date_range(test_start_date, test_end_date):
            date_str = date.strftime("%Y-%m-%d")
            L.info(f"test_all: {date_str}")

            # Buy
            for ticker_symbol in df_prices_dict.keys():
                df_prices = df_prices_dict[ticker_symbol]

                if len(df_prices.query(f"date=='{date_str}'")) == 0:
                    continue

                prices_id = df_prices.query(f"date=='{date_str}'").index[0]

                if df_prices.at[prices_id, "action"] != "trade":
                    continue

                buy_price = df_prices.at[prices_id, "open_price"]
                buy_stocks = init_asset * available_rate // buy_price

                if buy_stocks <= 0:
                    continue

                fee = buy_price * buy_stocks * fee_rate

                if (fund - buy_price * buy_stocks - fee) <= 0:
                    continue

                action_id = len(df_action)
                df_action.at[action_id, "date"] = date_str
                df_action.at[action_id, "ticker_symbol"] = ticker_symbol
                df_action.at[action_id, "action"] = "buy"
                df_action.at[action_id, "price"] = buy_price
                df_action.at[action_id, "stocks"] = buy_stocks
                df_action.at[action_id, "fee"] = fee

                stocks_id = len(df_stocks)
                df_stocks.at[stocks_id, "ticker_symbol"] = ticker_symbol
                df_stocks.at[stocks_id, "buy_price"] = buy_price
                df_stocks.at[stocks_id, "buy_stocks"] = buy_stocks
                df_stocks.at[stocks_id, "sell_id"] = df_prices.at[prices_id, "sell_id"]
                df_stocks.at[stocks_id, "sell_price"] = df_prices.at[prices_id, "sell_price"]

                fund = fund - buy_price * buy_stocks - fee

            # Sell
            for stocks_id in df_stocks.index:
                ticker_symbol = df_stocks.at[stocks_id, "ticker_symbol"]

                df_prices = df_prices_dict[ticker_symbol]

                if len(df_prices.query(f"date=='{date_str}'")) == 0:
                    continue

                prices_id = df_prices.query(f"date=='{date_str}'").index[0]

                if df_stocks.at[stocks_id, "sell_id"] != prices_id:
                    continue

                sell_price = df_stocks.at[stocks_id, "sell_price"]
                buy_price = df_stocks.at[stocks_id, "buy_price"]
                buy_stocks = df_stocks.at[stocks_id, "buy_stocks"]
                profit = (sell_price - buy_price) * buy_stocks
                profit_rate = profit / (sell_price * buy_stocks)
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

                df_stocks = df_stocks.drop(stocks_id)

                fund = fund + sell_price * buy_stocks - fee - tax

            df_stocks = df_stocks.assign(id=range(len(df_stocks)))
            df_stocks = df_stocks.set_index("id")

            # Update close_price_latest
            for stocks_id in df_stocks.index:
                ticker_symbol = df_stocks.at[stocks_id, "ticker_symbol"]

                df_prices = df_prices_dict[ticker_symbol]

                if len(df_prices.query(f"date=='{date_str}'")) == 0:
                    continue

                prices_id = df_prices.query(f"date=='{date_str}'").index[0]

                df_stocks.at[stocks_id, "close_price_latest"] = df_prices.at[prices_id, "close_price"]

            # Turn end
            asset = fund + (df_stocks["close_price_latest"] * df_stocks["buy_stocks"]).sum()

            df_result.at[date_str, "fund"] = fund
            df_result.at[date_str, "asset"] = asset

            L.info(df_result.loc[date_str])
            L.info(df_stocks)

        app_s3.write_dataframe(df_action, s3_bucket, f"{base_path}/test_all.action.csv")
        app_s3.write_dataframe(df_result, s3_bucket, f"{base_path}/test_all.result.csv")

        L.info("finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="simulate, forward_test, or forward_test_all")
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
            end_date="2019-01-01",
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/simulate_trade_4.{args.suffix}"
        )
    elif args.task == "forward_test":
        predictor = PredictClassification_3(
            job_name="predict_3",
            train_start_date=None,
            train_end_date=None,
            test_start_date=None,
            test_end_date=None,
            s3_bucket="u6k",
            input_preprocess_base_path=None,
            input_simulate_base_path=None,
            output_base_path=f"ml-data/stocks/predict_3.simulate_trade_4.{args.suffix}"
        )

        simulator.forward_test(
            s3_bucket="u6k",
            predictor=predictor,
            input_preprocess_base_path=f"ml-data/stocks/preprocess_3.{args.suffix}",
            input_simulate_base_path=f"ml-data/stocks/simulate_trade_4.{args.suffix}",
            output_base_path=f"ml-data/stocks/forward_test_4.{args.suffix}"
        )

        simulator.forward_test_report(
            start_date="2008-01-01",
            end_date="2018-01-01",
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/forward_test_4.{args.suffix}"
        )

        simulator.forward_test_report(
            start_date="2018-01-01",
            end_date="2019-01-01",
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/forward_test_4.{args.suffix}"
        )
    elif args.task == "forward_test_all":
        simulator.forward_test_all(
            report_start_date="2008-01-01",
            report_end_date="2018-01-01",
            test_start_date="2018-01-01",
            test_end_date="2019-01-01",
            s3_bucket="u6k",
            base_path=f"ml-data/stocks/forward_test_4.{args.suffix}"
        )
    else:
        parser.print_help()
