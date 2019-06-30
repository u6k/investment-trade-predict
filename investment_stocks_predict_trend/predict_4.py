from sklearn.linear_model import Lasso
from investment_stocks_predict_trend.predict_base import PredictRegressionBase


class PredictRegression_4(PredictRegressionBase):
    def execute(self):
        input_base_path = "local/predict_preprocessed"
        output_base_path = "local/predict_4"

        self.train(input_base_path, output_base_path)

    def preprocess(self, df_train_data, df_test_data, df_train_target, df_test_target):
        x_train = df_train_data.values
        x_test = df_test_data.values
        y_train = df_train_target["profit_rate"].values
        y_test = df_test_target["profit_rate"].values

        return x_train, x_test, y_train, y_test

    def model_fit(self, x_train, y_train):
        return Lasso().fit(x_train, y_train)
