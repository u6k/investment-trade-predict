import pandas as pd


def execute():
    input_base_path = "local/stock_prices_preprocessed"
    output_base_path = "local/simulate_trade_2"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    for ticker_symbol in df_companies.index:
        print(ticker_symbol)

        df_prices = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
        df_result = simulate_trade(df_prices)

        df_result.to_csv(f"{output_base_path}/result.{ticker_symbol}.csv")


def simulate_trade(df_prices_preprocessed, losscut_rate=0.95):
    df = df_prices_preprocessed.copy()

    # simulate
    for start_id in df.index:
        losscut_price = df.at[start_id, "close_price"] * losscut_rate
        end_id = None
        current_id = start_id + 1

        while df.index[-1] > current_id:
            if df.at[current_id, "close_price"] < losscut_price:
                end_id = current_id + 1
                break

            if losscut_price < (df.at[current_id, "close_price"] * losscut_rate):
                losscut_price = df.at[current_id, "close_price"] * losscut_rate

            current_id += 1

        if end_id is not None:
            df.at[start_id, "end_id"] = end_id
            df.at[start_id, "sell_price"] = df.at[end_id, "open_price"]
            df.at[start_id, "profit"] = df.at[end_id, "open_price"] - df.at[start_id, "close_price"]
            df.at[start_id, "profit_rate"] = df.at[end_id, "open_price"] / df.at[start_id, "close_price"]

    return df
