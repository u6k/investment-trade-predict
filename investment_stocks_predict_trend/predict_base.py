import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from app_logging import get_app_logger


class PredictClassificationBase():
    def execute(self):
        raise Exception("Not implemented.")

    def preprocess(self, df_data_train, df_data_test, df_target_train, df_target_test):
        raise Exception("Not implemented.")

    def model_fit(self, x_train, y_train):
        raise Exception("Not implemented.")

    def train(self, input_base_path, output_base_path):
        L = get_app_logger()
        L.info("start")

        df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
        df_result = pd.DataFrame(columns=df_companies.columns)

        results = joblib.Parallel(n_jobs=-1)([joblib.delayed(self.train_impl)(ticker_symbol, input_base_path, output_base_path) for ticker_symbol in df_companies.index])

        for result in results:
            ticker_symbol = result[0]
            scores = result[1]
            message = result[2]

            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]
            for key in scores.keys():
                df_result.at[ticker_symbol, key] = scores[key]
            df_result.at[ticker_symbol, "message"] = message

        df_result.to_csv(f"{output_base_path}/result.csv")
        L.info("finish")

    def train_impl(self, ticker_symbol, input_base_path, output_base_path):
        L = get_app_logger(ticker_symbol)
        L.info(f"train: {ticker_symbol}")

        try:
            x_train, x_test, y_train, y_test = self.load_data(input_base_path, ticker_symbol)

            clf = self.model_fit(x_train, y_train)
            joblib.dump(clf, f"{output_base_path}/model.{ticker_symbol}.joblib", compress=9)

            scores = self.model_score(clf, x_test, y_test)

            message = ""
        except Exception as err:
            L.exception(err)
            message = err.__str__()
            scores = {}

        return (ticker_symbol, scores, message)

    def load_data(self, input_base_path, ticker_symbol):
        # Load data
        df_data_train = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.data_train.csv", index_col=0)
        df_data_test = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.data_test.csv", index_col=0)
        df_target_train = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.target_train.csv", index_col=0)
        df_target_test = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.target_test.csv", index_col=0)

        return self.preprocess(df_data_train, df_data_test, df_target_train, df_target_test)

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


class PredictRegressionBase(PredictClassificationBase):
    def model_score(self, clf, x, y, df_result, result_id):
        y_pred = clf.predict(x)

        df_result.at[result_id, "rmse"] = np.sqrt(mean_squared_error(y, y_pred))
        df_result.at[result_id, "r2"] = r2_score(y, y_pred)

        df_predicted = pd.DataFrame()
        df_predicted["y_test"] = y
        df_predicted["y_pred"] = y_pred

        return df_predicted
