from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase


class SimulateTrade4(SimulateTradeBase):
    def simulate_singles_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(ticker_symbol)
        L.info(f"simulate_trade: {ticker_symbol}")

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

            message = ""
        except Exception as err:
            L.exception(err)
            message = err.__str__()

        return {
            "ticker_symbol": ticker_symbol,
            "message": message
        }


if __name__ == "__main__":
    s3_bucket = "u6k"
    input_base_path = "ml-data/stocks/preprocess_1.test"
    output_base_path = "ml-data/stocks/simulate_trade_4.test"

    SimulateTrade4().simulate_singles(s3_bucket, input_base_path, output_base_path)
