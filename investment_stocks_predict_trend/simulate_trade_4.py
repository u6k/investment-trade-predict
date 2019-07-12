import joblib
import pandas as pd

from app_logging import get_app_logger


def simulate_trade():
    L = get_app_logger()
    L.info("start")

    input_base_path = "local/preprocess_1"
    output_base_path = "local/simulate_trade_4"

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

    compare_high_price_period = 5
    hold_period = 5

    try:
        # Load data
        df = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

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
        df.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.csv")

        message = ""
    except Exception as err:
        L.exception(err)
        message = err.__str__()

    return (ticker_symbol, message)


if __name__ == "__main__":
    simulate_trade()
