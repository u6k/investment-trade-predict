import joblib
import numpy as np
import pandas as pd


class PredictClassificationBase():
    def execute(self):
        raise Exception("Not implemented.")

    def preprocess(self, df_train_data, df_test_data, df_train_target, df_test_target):
        raise Exception("Not implemented.")

    def model_fit(self, x_train, y_train):
        raise Exception("Not implemented.")

    def train(self, input_base_path, output_base_path):
        df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)

        df_result = pd.DataFrame(columns=df_companies.columns)

        for ticker_symbol in df_companies.query("message.isnull()").index:
            print(f"ticker_symbol={ticker_symbol}")

            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

            try:
                x_train, x_test, y_train, y_test = self.load_data(input_base_path, ticker_symbol)

                clf = self.model_fit(x_train, y_train)
                joblib.dump(clf, f"{output_base_path}/model.{ticker_symbol}.joblib", compress=9)

                self.model_score(clf, x_test, y_test, df_result, ticker_symbol)
            except Exception as err:
                print(err)
                df_result.at[ticker_symbol, "message"] = err.__str__()

            print(df_result.loc[ticker_symbol])
            df_result.to_csv(f"{output_base_path}/result.csv")

    def load_data(self, input_base_path, ticker_symbol):
        df_train_data = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.train_data.csv", index_col=0)
        df_test_data = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.test_data.csv", index_col=0)
        df_train_target = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.train_target.csv", index_col=0)
        df_test_target = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.test_target.csv", index_col=0)

        return self.preprocess(df_train_data, df_test_data, df_train_target, df_test_target)

    def model_score(self, clf, x, y, df_result, result_id):
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

        for label in labels:
            df_result.at[result_id, f"score_{label}_total"] = totals[label]
            df_result.at[result_id, f"score_{label}_count"] = counts[label]
            df_result.at[result_id, f"score_{label}"] = counts[label] / totals[label]
