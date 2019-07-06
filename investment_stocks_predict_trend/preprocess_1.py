import pandas as pd


def execute():
    input_base_path = "local/simulate_trade_2"
    output_base_path = "local/stock_prices_preprocessed"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies_result = pd.DataFrame(columns=df_companies.columns)

    for ticker_symbol in df_companies.index:
        print(f"ticker_symbol: {ticker_symbol}")

        df_companies_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        try:
            df_prices = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.simulated.csv", index_col=0)
            df_result = preprocess(df_prices)

            df_result.to_csv(f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            print(err)
            df_companies_result.at[ticker_symbol, "message"] = err.__str__()

        df_companies_result.to_csv(f"{output_base_path}/companies.csv")


def preprocess(df_prices):
    df = df_prices.copy()

    # Simple Moving Average
    for sma_len in [5, 10, 20, 40, 80]:
        df[f"sma_{sma_len}"] = df["adjusted_close_price"].rolling(sma_len).mean()

    # Momentum
    for momentum_len in [5, 10, 20, 40, 80]:
        df[f"momentum_{momentum_len}"] = df["adjusted_close_price"] - df["adjusted_close_price"].shift(momentum_len-1)

    # ROC (Rate Of Change)
    for roc_len in [5, 10, 20, 40, 80]:
        df[f"roc_{roc_len}"] = df["adjusted_close_price"].pct_change(roc_len-1)

    # RSI
    for rsi_len in [5, 10, 14, 20, 40]:
        diff = df["adjusted_close_price"].diff()
        diff = diff[1:]
        up, down = diff.copy(), diff.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        up_sma = up.rolling(window=rsi_len, center=False).mean()
        down_sma = down.rolling(window=rsi_len, center=False).mean()
        rsi = up_sma / (up_sma - down_sma) * 100.0

        df[f"rsi_{rsi_len}"] = rsi

    # Stochastic
    for stochastic_len in [5, 9, 20, 25, 40]:
        close = df["close_price"]
        low = df["low_price"]
        low_min = low.rolling(window=stochastic_len, center=False).min()
        high = df["high_price"]
        high_max = high.rolling(window=stochastic_len, center=False).max()

        stochastic_k = ((close - low_min) / (high_max - low_min)) * 100
        stochastic_d = stochastic_k.rolling(window=3, center=False).mean()
        stochastic_sd = stochastic_d.rolling(window=3, center=False).mean()

        df[f"stochastic_k_{stochastic_len}"] = stochastic_k
        df[f"stochastic_d_{stochastic_len}"] = stochastic_d
        df[f"stochastic_sd_{stochastic_len}"] = stochastic_sd

    return df
