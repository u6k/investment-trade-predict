import argparse

from sklearn.svm import OneClassSVM
from predict_base import PredictClassificationBase
import app_s3


class PredictClassification_8(PredictClassificationBase):
    def train_test_split(self, ticker_symbol, df):
        # Check data size
        if len(df.query(f"'{self._train_start_date}' <= date < '{self._train_end_date}'")) == 0 or len(df.query(f"'{self._test_start_date}' <= date < '{self._test_end_date}'")) == 0:
            raise Exception("little data")

        # Convert label
        for id in df.index:
            if df.at[id, "predict_target"] == 1:
                df.at[id, "predict_target"] = -1
            elif df.at[id, "predict_target"] == 0:
                df.at[id, "predict_target"] = 1

        # Split train/test
        train_start_id = df.query(f"'{self._train_start_date}' <= date < '{self._train_end_date}'").index[0]
        train_end_id = df.query(f"'{self._train_start_date}' <= date < '{self._train_end_date}'").index[-1]
        test_start_id = df.query(f"'{self._test_start_date}' <= date < '{self._test_end_date}'").index[0]
        test_end_id = df.query(f"'{self._test_start_date}' <= date < '{self._test_end_date}'").index[-1]

        df_data_train = df.loc[train_start_id: train_end_id].query("predict_target==1").drop(["date", "predict_target"], axis=1)
        df_data_test = df.loc[test_start_id: test_end_id].drop(["date", "predict_target"], axis=1)
        df_target_train = df.loc[train_start_id: train_end_id].query("predict_target==1")[["predict_target"]]
        df_target_test = df.loc[test_start_id: test_end_id][["predict_target"]]

        # Save data
        app_s3.write_dataframe(df_data_train, self._s3_bucket, f"{self._output_base_path}/stock_prices.{ticker_symbol}.data_train.csv")
        app_s3.write_dataframe(df_data_test, self._s3_bucket, f"{self._output_base_path}/stock_prices.{ticker_symbol}.data_test.csv")
        app_s3.write_dataframe(df_target_train, self._s3_bucket, f"{self._output_base_path}/stock_prices.{ticker_symbol}.target_train.csv")
        app_s3.write_dataframe(df_target_test, self._s3_bucket, f"{self._output_base_path}/stock_prices.{ticker_symbol}.target_test.csv")

        # Transform x, y
        x_train = df_data_train.values
        x_test = df_data_test.values
        y_train = df_target_train.values.flatten()
        y_test = df_target_test.values.flatten()

        return x_train, x_test, y_train, y_test

    def model_fit(self, x_train, y_train):
        return OneClassSVM(nu=0.003, kernel="rbf", gamma="auto").fit(x_train)

    def model_predict(self, ticker_symbol, df_data):
        model = app_s3.read_sklearn_model(self._s3_bucket, f"{self._output_base_path}/model.{ticker_symbol}.joblib")

        pred = model.predict(df_data.values)

        return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-group", help="simulate trade group")
    parser.add_argument("--suffix", help="folder name suffix (default: test)", default="test")
    args = parser.parse_args()

    pred = PredictClassification_8(
        job_name="predict_8",
        train_start_date="2008-01-01",
        train_end_date="2018-01-01",
        test_start_date="2018-01-01",
        test_end_date="2019-01-01",
        s3_bucket="u6k",
        input_preprocess_base_path=f"ml-data/stocks/preprocess_3.{args.suffix}",
        input_simulate_base_path=f"ml-data/stocks/simulate_trade_{args.simulate_group}.{args.suffix}",
        output_base_path=f"ml-data/stocks/predict_8.simulate_trade_{args.simulate_group}.{args.suffix}"
    )

    pred.train()
