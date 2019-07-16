import joblib

from app_logging import get_app_logger
import app_s3


def report_singles():
    L = get_app_logger()
    L.info("start")

    s3_bucket = "u6k"
    base_path = "ml-data/stocks/simulate_trade_3_backtest.20190713"

    df_companies = app_s3.read_dataframe(s3_bucket, f"{base_path}/companies.csv", index_col=0)

    results = joblib.Parallel(n_jobs=-1)([joblib.delayed(report)(s3_bucket, base_path, ticker_symbol) for ticker_symbol in df_companies.index])

    for result in results:
        ticker_symbol = result["ticker_symbol"]

        for k in result.keys():
            if k != "ticker_symbol":
                df_companies.at[ticker_symbol, k] = result[k]

    app_s3.write_dataframe(df_companies, s3_bucket, f"{base_path}/companies.report.csv")
    df_companies.to_csv("local/companies.simulate_trade_report.csv")
    L.info("finish")


def report(s3_bucket, base_path, ticker_symbol):
    L = get_app_logger(f"simulate_trade_4_report.{ticker_symbol}")
    L.info(f"report: {ticker_symbol}")

    result = {"ticker_symbol": ticker_symbol}

    try:
        df = app_s3.read_dataframe(s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        result["trade_count"] = len(df.query("not profit.isnull()"))
        result["win_count"] = len(df.query("profit>0"))
        result["win_rate"] = result["win_count"] / result["trade_count"]
        result["lose_count"] = len(df.query("profit<=0"))
        result["lose_rate"] = result["lose_count"] / result["trade_count"]
        result["open_price_latest"] = df["open_price"].values[-1]
        result["high_price_latest"] = df["high_price"].values[-1]
        result["low_price_latest"] = df["low_price"].values[-1]
        result["close_price_latest"] = df["close_price"].values[-1]
        result["volume_average"] = df["volume"].mean()
        result["expected_value"] = df["profit_rate"].mean()
        result["profit_total"] = df.query("profit>0")["profit"].sum()
        result["loss_total"] = df.query("profit<=0")["profit"].sum()
        result["profit_factor"] = result["profit_total"] / abs(result["loss_total"])
        result["profit_average"] = df.query("profit>0")["profit"].mean()
        result["loss_average"] = df.query("profit<=0")["profit"].mean()
        result["payoff_ratio"] = result["profit_average"] / abs(result["loss_average"])

        fund = 1000000
        for id in df.query("not profit.isnull()").index:
            df.at[id, "fund"] = fund + fund * df.at[id, "profit_rate"]
        for id in df.query("not profit.isnull()").index[: -1]:
            df.at[id, "min_fund"] = df.query(f"id > {id}")["fund"].min()
            df.at[id, "drawdown"] = (df.at[id, "fund"] - df.at[id, "min_fund"]) / df.at[id, "fund"]

        result["max_drawdown"] = df["drawdown"].max()

        result["message"] = ""
    except Exception as err:
        L.exception(err)
        result["message"] = err.__str__()

    return result


if __name__ == "__main__":
    report_singles()
