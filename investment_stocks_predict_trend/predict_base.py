import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE

from app_logging import get_app_logger
import app_s3


class PredictClassificationBase():
    def __init__(self, **kwargs):
        self._job_name = kwargs["job_name"] if "job_name" in kwargs else None
        self._train_start_date = kwargs["train_start_date"] if "train_start_date" in kwargs else None
        self._train_end_date = kwargs["train_end_date"] if "train_end_date" in kwargs else None
        self._test_start_date = kwargs["test_start_date"] if "test_start_date" in kwargs else None
        self._test_end_date = kwargs["test_end_date"] if "test_end_date" in kwargs else None
        self._s3_bucket = kwargs["s3_bucket"] if "s3_bucket" in kwargs else None
        self._input_preprocess_base_path = kwargs["input_preprocess_base_path"] if "input_preprocess_base_path" in kwargs else None
        self._input_simulate_base_path = kwargs["input_simulate_base_path"] if "input_simulate_base_path" in kwargs else None
        self._output_base_path = kwargs["output_base_path"] if "output_base_path" in kwargs else None

    def train(self):
        L = get_app_logger()
        L.info("start")

        df_companies = app_s3.read_dataframe(self._s3_bucket, f"{self._input_simulate_base_path}/companies.csv", index_col=0)
        df_result = pd.DataFrame(columns=df_companies.columns)

        results = joblib.Parallel(n_jobs=-1)([joblib.delayed(self.train_impl)(ticker_symbol) for ticker_symbol in df_companies.index])

        for result in results:
            if result["exception"] is not None:
                continue

            ticker_symbol = result["ticker_symbol"]
            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

            scores = result["scores_train"]
            for k in scores.keys():
                df_result.at[ticker_symbol, f"{k}_train"] = scores[k]

            scores = result["scores_test"]
            for k in scores.keys():
                df_result.at[ticker_symbol, f"{k}_test"] = scores[k]

        app_s3.write_dataframe(df_result, self._s3_bucket, f"{self._output_base_path}/report.csv")

        L.info("finish")

    def train_impl(self, ticker_symbol):
        L = get_app_logger()
        L.info(f"train: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None,
            "scores": None
        }

        try:
            df = self.preprocess(ticker_symbol)
            x_train, x_test, y_train, y_test = self.train_test_split(ticker_symbol, df)

            clf = self.model_fit(x_train, y_train)
            app_s3.write_sklearn_model(clf, self._s3_bucket, f"{self._output_base_path}/model.{ticker_symbol}.joblib")

            result["scores_train"] = self.model_score(clf, x_train, y_train)
            result["scores_test"] = self.model_score(clf, x_test, y_test)
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def preprocess(self, ticker_symbol):
        # Load data
        df_preprocess = app_s3.read_dataframe(self._s3_bucket, f"{self._input_preprocess_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
        df_simulate = app_s3.read_dataframe(self._s3_bucket, f"{self._input_simulate_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        # Preprocess
        df = df_preprocess
        df["predict_target"] = df_simulate[["predict_target_label"]]
        df = df.dropna()

        # Save data
        app_s3.write_dataframe(df, self._s3_bucket, f"{self._output_base_path}/stock_prices.{ticker_symbol}.csv")

        return df

    def train_test_split(self, ticker_symbol, df):
        # Check data size
        if len(df.query(f"'{self._train_start_date}' <= date < '{self._train_end_date}'")) == 0 or len(df.query(f"'{self._test_start_date}' <= date < '{self._test_end_date}'")) == 0:
            raise Exception("little data")

        # Split train/test
        train_start_id = df.query(f"'{self._train_start_date}' <= date < '{self._train_end_date}'").index[0]
        train_end_id = df.query(f"'{self._train_start_date}' <= date < '{self._train_end_date}'").index[-1]
        test_start_id = df.query(f"'{self._test_start_date}' <= date < '{self._test_end_date}'").index[0]
        test_end_id = df.query(f"'{self._test_start_date}' <= date < '{self._test_end_date}'").index[-1]

        df_data_train = df.loc[train_start_id: train_end_id].drop(["date", "predict_target"], axis=1)
        df_data_test = df.loc[test_start_id: test_end_id].drop(["date", "predict_target"], axis=1)
        df_target_train = df.loc[train_start_id: train_end_id][["predict_target"]]
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

        # Over sampling
        x_train, y_train = SMOTE().fit_sample(x_train, y_train)
        x_test, y_test = SMOTE().fit_sample(x_test, y_test)

        return x_train, x_test, y_train, y_test

    def model_fit(self, x_train, y_train):
        raise Exception("Not implemented.")

    def model_score(self, clf, x, y):
        totals = {}
        counts = {}

        labels = np.unique(y)

        for label in labels:
            totals[label] = 0
            counts[label] = 0

        y_pred = clf.predict(x)

        for i in range(len(y)):
            totals[y[i]] += 1
            if y[i] == y_pred[i]:
                counts[y[i]] += 1

        scores = {}
        for label in labels:
            scores[f"score_{label}_total"] = totals[label]
            scores[f"score_{label}_count"] = counts[label]
            scores[f"score_{label}"] = counts[label] / totals[label]

        return scores

    def model_predict(self, ticker_symbol, df_data):
        raise Exception("Not implemented.")


class PredictRegressionBase(PredictClassificationBase):
    def preprocess(self, ticker_symbol):
        # Load data
        df_preprocess = app_s3.read_dataframe(self._s3_bucket, f"{self._input_preprocess_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
        df_simulate = app_s3.read_dataframe(self._s3_bucket, f"{self._input_simulate_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        # Preprocess
        df = df_preprocess
        df["predict_target"] = df_simulate["predict_target_value"]
        df = df.dropna()

        # Save data
        app_s3.write_dataframe(df, self._s3_bucket, f"{self._output_base_path}/stock_prices.{ticker_symbol}.csv")

        return df

    def model_score(self, clf, x, y):
        y_pred = clf.predict(x)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        return {
            "rmse": rmse,
            "r2": r2
        }
