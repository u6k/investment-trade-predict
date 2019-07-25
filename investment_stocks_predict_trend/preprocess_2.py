import argparse
import joblib
import pandas as pd

from app_logging import get_app_logger
import app_s3


def execute(s3_bucket, input_base_path, output_base_path):
    L = get_app_logger("preprocess_2")
    L.info("start")

    df_companies = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/companies.csv", index_col=0)
    df_result = pd.DataFrame(columns=df_companies.columns)

    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(preprocess)(ticker_symbol, s3_bucket, input_base_path, output_base_path) for ticker_symbol in df_companies.index])

    for result in results:
        if result["exception"] is not None:
            continue

        ticker_symbol = result["ticker_symbol"]
        df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

    app_s3.write_dataframe(df_result, s3_bucket, f"{output_base_path}/companies.csv")

    L.info("finish")


def preprocess(ticker_symbol, s3_bucket, input_base_path, output_base_path):
    L = get_app_logger(f"preprocess_2.{ticker_symbol}")
    L.info(f"preprocess_2: {ticker_symbol}")

    result = {
        "ticker_symbol": ticker_symbol,
        "exception": None
    }

    try:
        df = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        # Volume change rate
        df["volume_change"] = df["volume"] / df["volume"].shift(1)

        # Adjusted close price change rate
        df["adjusted_close_price_change"] = df["adjusted_close_price"] / df["adjusted_close_price"].shift(1)

        # SMA (Simple Moving Average)
        sma_len_array = [5, 10, 20, 40, 80]
        for sma_len in sma_len_array:
            df[f"sma_{sma_len}"] = df["adjusted_close_price"].rolling(sma_len).mean()

        # Momentum
        momentum_len_array = [5, 10, 20, 40, 80]
        for momentum_len in momentum_len_array:
            df[f"momentum_{momentum_len}"] = df["adjusted_close_price"] - df["adjusted_close_price"].shift(momentum_len-1)

        # ROC (Rate Of Change)
        roc_len_array = [5, 10, 20, 40, 80]
        for roc_len in roc_len_array:
            df[f"roc_{roc_len}"] = df["adjusted_close_price"].pct_change(roc_len-1)

        # RSI
        rsi_len_array = [5, 10, 14, 20, 40]
        for rsi_len in rsi_len_array:
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
        stochastic_len_array = [5, 9, 20, 25, 40]
        for stochastic_len in stochastic_len_array:
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

        # Drop columns
        df = df.drop(["ticker_symbol", "open_price", "high_price", "low_price", "close_price", "volume", "adjusted_close_price"], axis=1)

        # Save
        app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
    except Exception as err:
        L.exception(f"ticker_symbol={ticker_symbol}, {err}")
        result["exception"] = err

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", help="folder name suffix (default: test)", default="test")
    args = parser.parse_args()

    execute(
        s3_bucket="u6k",
        input_base_path=f"ml-data/stocks/preprocess_1.{args.suffix}",
        output_base_path=f"ml-data/stocks/preprocess_2.{args.suffix}",
    )
