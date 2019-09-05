import argparse
from datetime import datetime

from app_logging import get_app_logger
from test_trade_base import TradeSimulaterBase


class TradeSimulater1(TradeSimulaterBase):
    def technical_trade(self, df):
        L = get_app_logger()
        L.info("technical_trade")

        df["buy_section"] = 1

        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    TradeSimulater1().execute(
        fxpairs=["usdjpy"],
        start_date=datetime(2018, 1, 1),
        end_date=datetime(2018, 3, 1),
        input_explanatory_base_path="preprocess_3",
        input_explained_base_path="preprocess_4",
        input_model_path="predict_1/model.usdjpy.20180101_20180201.joblib",
        output_base_path=f"test_1",
    )
