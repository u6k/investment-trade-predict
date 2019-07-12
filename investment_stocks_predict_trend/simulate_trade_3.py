import joblib
import pandas as pd

from app_logging import get_app_logger


def simulate_trade():
    L = get_app_logger()
    L.info("start")

    input_base_path = "local/preprocess_1"
    output_base_path = "local/simulate_trade_3"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(simulate_trade_impl)(ticker_symbol, input_base_path, output_base_path) for ticker_symbol in df_companies.index])

    for result in results:
        ticker_symbol = result[0]
        message = result[1]

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]
        df_companies_result.at[ticker_symbol, "message"] = message

    df_companies_result.to_csv(f"{output_base_path}/companies.csv")
    L.info("finish")


def simulate_trade_impl(ticker_symbol, input_base_path, output_base_path):
    L = get_app_logger(ticker_symbol)
    L.info(f"simulate_trade: {ticker_symbol}")

    try:
        df = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        df["day_trade_profit_rate"] = df["close_price"] / df["open_price"]
        df["day_trade_profit_flag"] = df["day_trade_profit_rate"].apply(lambda r: 1 if r > 1.0 else 0)

        df.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.csv")

        message = ""
    except Exception as err:
        L.exception(err)
        message = err.__str__()

    return (ticker_symbol, message)


if __name__ == "__main__":
    simulate_trade()
