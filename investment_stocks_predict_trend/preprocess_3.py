import pandas as pd


def execute():
    input_base_path = "local/stock_prices_preprocessed"
    output_base_path = "local/stock_prices_preprocessed_3"

    # Load companies
    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    for ticker_symbol in df_companies.index:
        print(ticker_symbol)

        # Load data
        df_prices = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        # Preprocess
        df_prices_result = preprocess_day_trade(df_prices)

        # Save result
        df_prices_result.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        df_companies_result.to_csv(f"{output_base_path}/companies.csv")

        print(df_companies_result.loc[ticker_symbol])


def preprocess_day_trade(df_prices):
    df = df_prices.copy()

    df["day_trade_profit_rate"] = df["close_price"] / df["open_price"]
    df["day_trade_profit_flag"] = df["day_trade_profit_rate"].apply(lambda r: 1 if r > 1.0 else 0)

    return df
