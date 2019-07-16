from app_logging import get_app_logger
import app_s3
from simulate_trade_base import SimulateTradeBase


class SimulateTrade2(SimulateTradeBase):
    def simulate_singles_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(ticker_symbol)
        L.info(f"simulate_trade: {ticker_symbol}")

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
                    df.at[start_id, "sell_price"] = df.at[end_id, "low_price"]
                    df.at[start_id, "profit"] = df.at[end_id, "low_price"] - df.at[start_id, "open_price"]
                    df.at[start_id, "profit_rate"] = df.at[end_id, "low_price"] / df.at[start_id, "open_price"]

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
    output_base_path = "ml-data/stocks/simulate_trade_2.test"

    SimulateTrade2().simulate_singles(s3_bucket, input_base_path, output_base_path)
