import joblib
import pandas as pd

from app_logging import get_app_logger
import app_s3


def execute():
    L = get_app_logger()
    L.info("start")

    s3_bucket = "u6k"
    input_base_path = "ml-data/stocks/preprocess_1.test"
    output_base_path = "ml-data/stocks/simulate_trade_2.test"

    df_companies = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(simulate_trade)(ticker_symbol, s3_bucket, input_base_path, output_base_path) for ticker_symbol in df_companies.index])

    for result in results:
        ticker_symbol = result["ticker_symbol"]

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]
        df_companies_result.at[ticker_symbol, "message"] = result["message"]

    app_s3.write_dataframe(df_companies_result, s3_bucket, f"{output_base_path}/companies.csv")
    df_companies_result.to_csv("companies.simulate_trade_2.csv")
    L.info("finish")


def simulate_trade(ticker_symbol, s3_bucket, input_base_path, output_base_path, losscut_rate=0.95):
    L = get_app_logger(ticker_symbol)
    L.info(f"simulate_trade: {ticker_symbol}")

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
    execute()
