import joblib

from app_logging import get_app_logger
import app_s3


class SimulateTradeBase():
    def simulate_singles(self, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger()
        L.info("start")

        df_companies = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/companies.csv", index_col=0)

        results = joblib.Parallel(n_jobs=-1)([joblib.delayed(self.simulate_singles_impl)(ticker_symbol, s3_bucket, input_base_path, output_base_path) for ticker_symbol in df_companies.index])

        for result in results:
            ticker_symbol = result["ticker_symbol"]

            for k in result.keys():
                if k != "ticker_symbol":
                    df_companies.at[ticker_symbol, k] = result[k]

        app_s3.write_dataframe(df_companies, s3_bucket, f"{output_base_path}/companies.csv")
        L.info("finish")

    def simulate_singles_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        raise Exception("Not implemented.")
