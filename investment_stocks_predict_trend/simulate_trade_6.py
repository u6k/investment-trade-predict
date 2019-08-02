import argparse

from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase
# from predict_3 import PredictClassification_3
from predict_7 import PredictClassification_7


class SimulateTrade6(SimulateTradeBase):
    def simulate_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(f"{self._job_name}.simulate_impl.{ticker_symbol}")
        L.info(f"{self._job_name}.simulate_impl: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        ema_len_array = [9, 26]
        losscut_rate = 0.98
        take_profit_rate = 0.95

        minimum_profit_rate = 0.03

        try:
            # Load data
            df = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            # Preprocess
            for ema_len in ema_len_array:
                df[f"ema_{ema_len}"] = df["adjusted_close_price"].ewm(ema_len).mean()
                df[f"ema_{ema_len}_1"] = df[f"ema_{ema_len}"].shift(1)

            # Set signal
            target_id_array = df.query(f"(ema_{ema_len_array[0]}_1 < ema_{ema_len_array[1]}_1) and (ema_{ema_len_array[0]} >= ema_{ema_len_array[1]})").index
            for id in target_id_array:
                df.at[id, "signal"] = "buy"

            target_id_array = df.query(f"(ema_{ema_len_array[0]}_1 > ema_{ema_len_array[1]}_1) and (ema_{ema_len_array[0]} <= ema_{ema_len_array[1]})").index
            for id in target_id_array:
                df.at[id, "signal"] = "sell"

            # Simulate
            buy_id = None
            losscut_price = None
            take_profit_price = None
            take_profit = None

            for id in df.index[1:]:
                # Sell: take profit
                if take_profit:
                    buy_price = df.at[buy_id, "open_price"]
                    sell_price = df.at[id, "open_price"]
                    profit = sell_price - buy_price
                    profit_rate = profit / sell_price

                    df.at[buy_id, "result"] = "take profit"
                    df.at[buy_id, "buy_date"] = df.at[buy_id, "date"]
                    df.at[buy_id, "buy_price"] = buy_price
                    df.at[buy_id, "sell_date"] = df.at[id, "date"]
                    df.at[buy_id, "sell_price"] = sell_price
                    df.at[buy_id, "profit"] = profit
                    df.at[buy_id, "profit_rate"] = profit_rate

                    buy_id = None
                    losscut_price = None
                    take_profit_price = None
                    take_profit = None

                # Sell: losscut
                if buy_id is not None and df.at[id, "low_price"] < losscut_price:
                    buy_price = df.at[buy_id, "open_price"]
                    sell_price = df.at[id, "open_price"]
                    profit = sell_price - buy_price
                    profit_rate = profit / sell_price

                    df.at[buy_id, "result"] = "losscut"
                    df.at[buy_id, "buy_date"] = df.at[buy_id, "date"]
                    df.at[buy_id, "buy_price"] = buy_price
                    df.at[buy_id, "sell_date"] = df.at[id, "date"]
                    df.at[buy_id, "sell_price"] = sell_price
                    df.at[buy_id, "profit"] = profit
                    df.at[buy_id, "profit_rate"] = profit_rate

                    buy_id = None
                    losscut_price = None
                    take_profit_price = None
                    take_profit = None

                # Flag: take profit
                if buy_id is not None and df.at[id, "high_price"] < take_profit_price:
                    take_profit = True

                # Buy
                if buy_id is None and df.at[id-1, "signal"] == "buy":
                    buy_id = id
                    losscut_price = df.at[id, "close_price"] * losscut_rate
                    take_profit_price = df.at[id, "high_price"] * take_profit_rate
                    take_profit = False

                # Sell
                if buy_id is not None and df.at[id-1, "signal"] == "sell":
                    buy_price = df.at[buy_id, "open_price"]
                    sell_price = df.at[id, "open_price"]
                    profit = sell_price - buy_price
                    profit_rate = profit / sell_price

                    df.at[buy_id, "result"] = "sell signal"
                    df.at[buy_id, "buy_date"] = df.at[buy_id, "date"]
                    df.at[buy_id, "buy_price"] = buy_price
                    df.at[buy_id, "sell_date"] = df.at[id, "date"]
                    df.at[buy_id, "sell_price"] = sell_price
                    df.at[buy_id, "profit"] = profit
                    df.at[buy_id, "profit_rate"] = profit_rate

                    buy_id = None
                    losscut_price = None
                    take_profit_price = None
                    take_profit = None

                # Update losscut/take profit price
                if buy_id is not None:
                    losscut_price_tmp = df.at[id, "close_price"] * losscut_rate
                    if losscut_price_tmp > losscut_price:
                        losscut_price = losscut_price_tmp

                    take_profit_price_tmp = df.at[id, "high_price"] * take_profit_rate
                    if take_profit_price_tmp > take_profit_price:
                        take_profit_price = take_profit_price_tmp

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

    simulator = SimulateTrade6("simulate_trade_6")

    s3_bucket = "u6k"

    predictor_name = "predict_7"
    predictor = PredictClassification_7(
        job_name=predictor_name,
        s3_bucket=s3_bucket,
        output_base_path=f"ml-data/stocks/{predictor_name}.simulate_trade_6.{args.suffix}"
    )

    simulate_input_base_path = f"ml-data/stocks/preprocess_1.{args.suffix}"
    simulate_output_base_path = f"ml-data/stocks/simulate_trade_6.{args.suffix}"
    test_preprocess_base_path = f"ml-data/stocks/preprocess_3.{args.suffix}"
    test_output_base_path = f"ml-data/stocks/forward_test_6.{predictor_name}.{args.suffix}"

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
