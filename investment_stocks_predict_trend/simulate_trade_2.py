import pandas as pd


def execute():
    input_base_path = "local/preprocess_1"
    output_base_path = "local/simulate_trade_2"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    for ticker_symbol in df_companies.index:
        print(f"ticker_symbol={ticker_symbol}")

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        try:
            simulate_trade(ticker_symbol, input_base_path, output_base_path)
            df_companies_result.at[ticker_symbol, "message"] = ""
        except Exception as err:
            df_companies_result.at[ticker_symbol, "message"] = err.__str__()

        df_companies_result.to_csv(f"{output_base_path}/companies.csv")
        print(df_companies_result.loc[ticker_symbol])


def simulate_trade(ticker_symbol, input_base_path, output_base_path, losscut_rate=0.95):
    df = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

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

    df.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.csv")


if __name__ == "__main__":
    execute()
