import pandas as pd
import numpy as np

from app_logging import get_app_logger
import app_s3


class TradeSimulaterBase():
    def execute(self, fxpairs, start_date, end_date, input_explanatory_base_path, input_explained_base_path, input_model_path, output_base_path):
        L = get_app_logger()
        L.info(f"start: fxpairs={fxpairs}, start_date={start_date}, end_date={end_date}")

        # Load data
        df_fxpairs = app_s3.read_dataframe(
            f"{input_explained_base_path}/fxpairs.csv",
            dtype={"id": int, "fxpair": str, "start_date": str, "end_date": str},
            parse_dates=["start_date", "end_date"],
            index_col=0
        )
        df_result = pd.DataFrame(columns=df_fxpairs.columns)

        # Trade test
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        for fxpair in fxpairs:
            for id in df_fxpairs.query(f"fxpair=='{fxpair}' and start_date>='{start_date_str}' and start_date<'{end_date_str}'").index:
                fxpair = df_fxpairs.at[id, "fxpair"]
                test_start_date = df_fxpairs.at[id, "start_date"]
                test_end_date = df_fxpairs.at[id, "end_date"]

                # Trade
                df = self.preprocess(fxpair, test_start_date, test_end_date, input_explained_base_path)

                df = self.predict_trade(fxpair, test_start_date, test_end_date, input_explanatory_base_path, input_model_path, df)

                df = self.technical_trade(df)

                df = df.dropna()

                # Report
                result = self.report_trade(df)

                df_result.loc[id] = df_fxpairs.loc[id]
                for k in result.keys():
                    df_result.at[id, k] = result[k]

                L.info(f"result: {df_result.loc[id]}")

                # Save
                app_s3.write_dataframe(df, f"{output_base_path}/fx.{test_start_date.strftime('%Y%m%d')}_{test_end_date.strftime('%Y%m%d')}.csv")
                app_s3.write_dataframe(df_result, f"{output_base_path}/report.csv")

        L.info("finish")

    def preprocess(self, fxpair, start_date, end_date, input_explained_base_path):
        L = get_app_logger()
        L.info(f"preprocess: fxpair={fxpair}, start_date={start_date}, end_date={end_date}")

        # Load data
        df = app_s3.read_dataframe(f"{input_explained_base_path}/fx.{fxpair}.{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv", index_col=0)

        return df

    def predict_trade(self, fxpair, start_date, end_date, input_explanatory_base_path, input_model_path, df):
        L = get_app_logger()
        L.info(f"predict_trade: fxpair={fxpair}, start_date={start_date}, end_date={end_date}")

        # Predict
        df_explanatory = app_s3.read_dataframe(f"{input_explanatory_base_path}/fx.{fxpair}.{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv", index_col=0)

        df_test = df_explanatory.drop("timestamp", axis=1).dropna()
        model = app_s3.read_sklearn_model(input_model_path)
        df_test["predict"] = model.predict(df_test.values)

        df["predict"] = df_test["predict"]

        return df

    def technical_trade(self, df):
        raise Exception("Not implemented.")

    def report_trade(self, df):
        L = get_app_logger()
        L.info("report_trade")

        df_target = df.query("predict==1 and buy_section==1 and not profit.isnull()")
        df_target_win_only = df_target.query("profit>0")
        df_target_lose_only = df_target.query("profit<=0")

        result = {
            "trade_count": len(df_target)
        }

        if result["trade_count"] == 0:
            return result

        result["win_count"] = len(df_target_win_only)
        result["win_rate"] = result["win_count"] / result["trade_count"]
        result["lose_count"] = len(df_target_lose_only)
        result["lose_rate"] = result["lose_count"] / result["trade_count"]
        result["expected_value"] = df_target["profit_rate"].mean()
        result["expected_value_win_only"] = df_target_win_only["profit_rate"].mean()
        result["risk"] = df_target["profit_rate"].std()
        result["profit_total"] = df_target_win_only["profit"].sum()
        result["loss_total"] = df_target_lose_only["profit"].sum()
        if result["loss_total"] == 0:
            result["profit_factor"] = np.inf
        else:
            result["profit_factor"] = result["profit_total"] / abs(result["loss_total"])
        result["profit_average"] = df_target_win_only["profit"].mean()
        result["loss_average"] = df_target_lose_only["profit"].mean()
        if result["loss_average"] == 0:
            result["payoff_ratio"] = np.inf
        else:
            result["payoff_ratio"] = result["profit_average"] / abs(result["loss_average"])
        result["sharpe_ratio"] = result["expected_value"] / result["risk"]
        result["profit_rate"] = (df_target["sell_price"].sum() - df_target["buy_price"].sum()) / df_target["sell_price"].sum()
        result["profit_rate_win_only"] = (df_target_win_only["sell_price"].sum() - df_target_win_only["buy_price"].sum()) / df_target_win_only["sell_price"].sum()

        return result
