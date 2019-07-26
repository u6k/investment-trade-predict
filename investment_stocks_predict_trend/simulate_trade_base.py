from datetime import timedelta
import joblib
import pandas as pd

from app_logging import get_app_logger
import app_s3


class SimulateTradeBase():
    def __init__(self, job_name):
        self._job_name = job_name

    def simulate(self, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(f"{self._job_name}.simulate")
        L.info(f"{self._job_name}.simulate: start")

        df_companies = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/companies.csv", index_col=0)
        df_result = pd.DataFrame(columns=df_companies.columns)

        results = joblib.Parallel(n_jobs=-1)([joblib.delayed(self.simulate_impl)(ticker_symbol, s3_bucket, input_base_path, output_base_path) for ticker_symbol in df_companies.index])

        for result in results:
            if result["exception"] is not None:
                continue

            ticker_symbol = result["ticker_symbol"]
            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        app_s3.write_dataframe(df_result, s3_bucket, f"{output_base_path}/companies.csv")

        L.info(f"{self._job_name}.simulate: finish")

    def simulate_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        raise Exception("Not implemented.")

    def simulate_report(self, start_date, end_date, s3_bucket, base_path):
        L = get_app_logger(f"{self._job_name}.simulate_report")
        L.info(f"{self._job_name}.simulate_report: start")

        df_companies = app_s3.read_dataframe(s3_bucket, f"{base_path}/companies.csv", index_col=0)
        df_result = pd.DataFrame(columns=df_companies.columns)

        results = joblib.Parallel(n_jobs=-1)([joblib.delayed(self.report_impl)(ticker_symbol, start_date, end_date, s3_bucket, base_path) for ticker_symbol in df_companies.index])

        for result in results:
            if result["exception"] is not None:
                continue

            ticker_symbol = result["ticker_symbol"]
            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

            for k in result.keys():
                if k != "ticker_symbol" and k != "exception":
                    df_result.at[ticker_symbol, k] = result[k]

        app_s3.write_dataframe(df_result, s3_bucket, f"{base_path}/report.csv")
        L.info("finish")

    def forward_test(self, start_date, end_date, s3_bucket, input_simulate_base_path, input_model_base_path, output_base_path):
        L = get_app_logger(f"{self._job_name}.forward_test")
        L.info(f"{self._job_name}.forward_test: start")

        df_companies = app_s3.read_dataframe(s3_bucket, f"{input_simulate_base_path}/companies.csv", index_col=0)
        df_report = app_s3.read_dataframe("u6k", f"{input_model_base_path}/report.csv", index_col=0)
        df_result = pd.DataFrame(columns=df_companies.columns)

        results = joblib.Parallel(n_jobs=-1)([joblib.delayed(self.forward_test_impl)(ticker_symbol, start_date, end_date, s3_bucket, input_simulate_base_path, input_model_base_path, output_base_path) for ticker_symbol in df_report.index])

        for result in results:
            if result["exception"] is not None:
                continue

            ticker_symbol = result["ticker_symbol"]
            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        app_s3.write_dataframe(df_result, s3_bucket, f"{output_base_path}/companies.csv")
        L.info("finish")

    def forward_test_impl(self, ticker_symbol, start_date, end_date, s3_bucket, input_simulate_base_path, input_model_base_path, output_base_path):
        raise Exception("Not implemented.")

    def forward_test_report(self, start_date, end_date, s3_bucket, base_path):
        L = get_app_logger(f"{self._job_name}.forward_test_report")
        L.info(f"{self._job_name}.forward_test_report: start")

        df_companies = app_s3.read_dataframe(s3_bucket, f"{base_path}/companies.csv", index_col=0)
        df_result = pd.DataFrame(columns=df_companies.columns)

        results = joblib.Parallel(n_jobs=-1)([joblib.delayed(self.report_impl)(ticker_symbol, start_date, end_date, s3_bucket, base_path) for ticker_symbol in df_companies.index])

        for result in results:
            if result["exception"] is not None:
                continue

            ticker_symbol = result["ticker_symbol"]
            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

            for k in result.keys():
                if k != "ticker_symbol" and k != "exception":
                    df_result.at[ticker_symbol, k] = result[k]

        app_s3.write_dataframe(df_result, s3_bucket, f"{base_path}/report.csv")
        L.info("finish")

    def report_impl(self, ticker_symbol, start_date, end_date, s3_bucket, base_path):
        L = get_app_logger(f"{self._job_name}.report_impl.{ticker_symbol}")
        L.info(f"{self._job_name}.report_impl: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        try:
            df = app_s3.read_dataframe(s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0) \
                .query(f"'{start_date}' <= date <= '{end_date}'")

            if len(df) == 0 or "profit" not in df.columns:
                raise Exception("no trade")

            result["trade_count"] = len(df.query("not profit.isnull()"))
            result["win_count"] = len(df.query("profit>0"))
            result["win_rate"] = result["win_count"] / result["trade_count"] if result["trade_count"] != 0 else None
            result["lose_count"] = len(df.query("profit<=0"))
            result["lose_rate"] = result["lose_count"] / result["trade_count"] if result["trade_count"] != 0 else None
            result["open_price_latest"] = df["open_price"].values[-1]
            result["high_price_latest"] = df["high_price"].values[-1]
            result["low_price_latest"] = df["low_price"].values[-1]
            result["close_price_latest"] = df["close_price"].values[-1]
            result["volume_average"] = df["volume"].mean()
            result["expected_value"] = df["profit_rate"].mean()
            result["expected_value_win_only"] = df.query("profit>0")["profit_rate"].mean()
            result["risk"] = df["profit_rate"].std()
            result["profit_total"] = df.query("profit>0")["profit"].sum()
            result["loss_total"] = df.query("profit<=0")["profit"].sum()
            result["profit_factor"] = result["profit_total"] / abs(result["loss_total"]) if result["loss_total"] != 0 else None
            result["profit_average"] = df.query("profit>0")["profit"].mean()
            result["loss_average"] = df.query("profit<=0")["profit"].mean()
            result["payoff_ratio"] = result["profit_average"] / abs(result["loss_average"]) if result["loss_average"] != 0 else None
            result["sharpe_ratio"] = result["expected_value"] / result["risk"] if result["risk"] != 0 else None
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def date_range(self, start, end):
        for n in range((end - start).days):
            yield start + timedelta(n)
