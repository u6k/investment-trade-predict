import argparse

from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase
from predict_5 import PredictClassification_5


class SimulateTrade2(SimulateTradeBase):
    def simulate_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger()
        L.info(f"{self._job_name}.simulate_impl: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        losscut_rate = 0.98
        take_profit_rate = 0.95
        minimum_profit_rate = 0.03

        try:
            # Load data
            df = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            for column in df.columns:
                if column.startswith("index_"):
                    df = df.drop(column, axis=1)

            # Simulate
            for buy_id in df.index:
                buy_price = df.at[buy_id, "open_price"]
                losscut_price = buy_price * losscut_rate
                take_profit_price = buy_price * take_profit_rate

                sell_id = None
                sell_price = None
                take_profit = False

                for id in df.loc[buy_id+1:].index:
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

                    # Flag take profit
                    if df.at[id, "high_price"] < take_profit_price:
                        take_profit = True

                    # Update losscut/take profit price
                    if losscut_price < (df.at[id, "close_price"] * losscut_rate):
                        losscut_price = df.at[id, "close_price"] * losscut_rate
                    if take_profit_price < (df.at[id, "high_price"] * take_profit_rate):
                        take_profit_price = df.at[id, "high_price"] * take_profit_rate

                # Set result
                if sell_id is not None:
                    df.at[buy_id, "buy_date"] = df.at[buy_id, "date"]
                    df.at[buy_id, "buy_price"] = buy_price
                    df.at[buy_id, "sell_date"] = df.at[sell_id, "date"]
                    df.at[buy_id, "sell_price"] = sell_price
                    df.at[buy_id, "profit"] = sell_price - buy_price
                    df.at[buy_id, "profit_rate"] = df.at[buy_id, "profit"] / sell_price

            # Labeling for predict
            df["predict_target_value"] = df["profit_rate"].shift(-1)
            df["predict_target_label"] = df["predict_target_value"].apply(lambda r: 1 if r >= minimum_profit_rate else 0)

            # Save data
            app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="simulate, forward_test, or forward_test_all")
    parser.add_argument("--suffix", help="folder name suffix (default: test)", default="test")
    args = parser.parse_args()

    simulator = SimulateTrade2("simulate_trade_2")

    s3_bucket = "u6k"

    predictor_name = "predict_5"
    predictor = PredictClassification_5(
        job_name=predictor_name,
        s3_bucket=s3_bucket,
        output_base_path=f"ml-data/stocks/{predictor_name}.simulate_trade_2.{args.suffix}"
    )

    simulate_input_base_path = f"ml-data/stocks/preprocess_2.{args.suffix}"
    simulate_output_base_path = f"ml-data/stocks/simulate_trade_2.{args.suffix}"
    test_preprocess_base_path = f"ml-data/stocks/preprocess_3.{args.suffix}"
    test_output_base_path = f"ml-data/stocks/forward_test_2.{predictor_name}.{args.suffix}"

    report_start_date = "2008-01-01"
    report_end_date = "2018-01-01"
    simulate_start_date = "2018-01-01"
    simulate_end_date = "2019-01-01"

    if args.task == "simulate":
        simulator.simulate(s3_bucket=s3_bucket, input_base_path=simulate_input_base_path, output_base_path=simulate_output_base_path)
        simulator.simulate_report(start_date=simulate_start_date, end_date=simulate_end_date, s3_bucket=s3_bucket, base_path=simulate_output_base_path)
    elif args.task == "forward_test":
        simulator.forward_test(s3_bucket=s3_bucket, predictor=predictor, input_preprocess_base_path=test_preprocess_base_path, input_simulate_base_path=simulate_output_base_path, output_base_path=test_output_base_path)
        simulator.forward_test_report(start_date=report_start_date, end_date=report_end_date, s3_bucket=s3_bucket, base_path=test_output_base_path)
        simulator.forward_test_report(start_date=simulate_start_date, end_date=simulate_end_date, s3_bucket=s3_bucket, base_path=test_output_base_path)
    elif args.task == "forward_test_all":
        simulator.forward_test_all(report_start_date=report_start_date, report_end_date=report_end_date, test_start_date=simulate_start_date, test_end_date=simulate_end_date, s3_bucket=s3_bucket, base_path=test_output_base_path)
    else:
        parser.print_help()
