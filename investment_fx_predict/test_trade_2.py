import argparse
from datetime import datetime

from app_logging import get_app_logger
from test_trade_base import TradeSimulaterBase


class TradeSimulater2(TradeSimulaterBase):
    def technical_trade(self, df):
        L = get_app_logger()
        L.info("technical_trade")

        # EMA (Exponential Moving Average)
        ema_len_short = 9
        ema_len_long = 26

        for ema_len in [ema_len_short, ema_len_long]:
            df[f"ema_{ema_len}"] = df["close"].ewm(span=ema_len).mean()

        df[f"ema_{ema_len_short}_1"] = df[f"ema_{ema_len_short}"].shift(1)
        df[f"ema_{ema_len_long}_1"] = df[f"ema_{ema_len_long}"].shift(1)

        # Set signal
        for id in df.query(f"(ema_{ema_len_short}_1 < ema_{ema_len_long}_1) and (ema_{ema_len_short} >= ema_{ema_len_long}) and (ema_{ema_len_long}_1 < ema_{ema_len_long})").index:
            df.at[id, "signal"] = "buy"

        for id in df.query(f"(ema_{ema_len_short}_1 > ema_{ema_len_long}_1) and (ema_{ema_len_short} <= ema_{ema_len_long}) and (ema_{ema_len_long}_1 > ema_{ema_len_long})").index:
            df.at[id, "signal"] = "sell"

        for id in df[1:].query("signal.isnull()").index:
            df.at[id, "signal"] = df.at[id-1, "signal"]

        buy_section = False
        df["buy_section"] = 0

        for id in df.index[1:]:
            if df.at[id-1, "signal"] == "buy":
                buy_section = True

            if df.at[id-1, "signal"] == "sell":
                buy_section = False

            if buy_section:
                df.at[id, "buy_section"] = 1

        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    TradeSimulater2().execute(
        fxpairs=["usdjpy"],
        start_date=datetime(2018, 1, 1),
        end_date=datetime(2018, 3, 1),
        input_explanatory_base_path="preprocess_3",
        input_explained_base_path="preprocess_4",
        input_model_path="predict_1/model.usdjpy.20180101_20180201.joblib",
        output_base_path=f"test_2",
    )
