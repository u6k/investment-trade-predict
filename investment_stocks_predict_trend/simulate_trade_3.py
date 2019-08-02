import argparse

from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase
# from predict_3 import PredictClassification_3
from predict_7 import PredictClassification_7


class SimulateTrade3(SimulateTradeBase):
    def simulate_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(f"{self._job_name}.simulate_impl.{ticker_symbol}")
        L.info(f"{self._job_name}.simulate_impl: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        minimum_profit_rate = 0.03

        try:
            # Load data
            df = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            # Simulate
            df["buy_date"] = df["date"]
            df["buy_price"] = df["open_price"]
            df["sell_date"] = df["date"]
            df["sell_price"] = df["close_price"]
            df["profit"] = df["sell_price"] - df["buy_price"]
            df["profit_rate"] = df["profit"] / df["sell_price"]

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

    simulator = SimulateTrade3("simulate_trade_3")

    s3_bucket = "u6k"

    predictor_name = "predict_7"
    predictor = PredictClassification_7(
        job_name=predictor_name,
        s3_bucket=s3_bucket,
        output_base_path=f"ml-data/stocks/{predictor_name}.simulate_trade_3.{args.suffix}"
    )

    simulate_input_base_path = f"ml-data/stocks/preprocess_1.{args.suffix}"
    simulate_output_base_path = f"ml-data/stocks/simulate_trade_3.{args.suffix}"
    test_preprocess_base_path = f"ml-data/stocks/preprocess_3.{args.suffix}"
    test_output_base_path = f"ml-data/stocks/forward_test_3.{predictor_name}.{args.suffix}"

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
