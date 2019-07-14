from sklearn.linear_model import Lasso
from predict_base import PredictRegressionBase


class PredictRegression_4(PredictRegressionBase):
    def execute(self):
        s3_bucket = "u6k"
        input_base_path = "ml-data/stocks/preprocess_5.test"
        output_base_path = "ml-data/stocks/predict_4_preprocess_5.test"

        self.train(s3_bucket, input_base_path, output_base_path)

    def preprocess(self, df_data_train, df_data_test, df_target_train, df_target_test):
        x_train = df_data_train.values
        x_test = df_data_test.values
        y_train = df_target_train["predict_target_value"].values
        y_test = df_target_test["predict_target_value"].values

        return x_train, x_test, y_train, y_test

    def model_fit(self, x_train, y_train):
        return Lasso().fit(x_train, y_train)


if __name__ == "__main__":
    PredictRegression_4().execute()
