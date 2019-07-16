from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase


class SimulateTrade3(SimulateTradeBase):
    def simulate_singles_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(ticker_symbol)
        L.info(f"simulate_trade: {ticker_symbol}")

        try:
            df = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            df["day_trade_profit_rate"] = df["close_price"] / df["open_price"]
            df["day_trade_profit_flag"] = df["day_trade_profit_rate"].apply(lambda r: 1 if r > 1.0 else 0)

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
    output_base_path = "ml-data/stocks/simulate_trade_3.test"

    SimulateTrade3().simulate_singles(s3_bucket, input_base_path, output_base_path)
