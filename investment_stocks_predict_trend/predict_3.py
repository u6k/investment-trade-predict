from sklearn import ensemble
from investment_stocks_predict_trend.predict_base import PredictClassificationBase


class PredictClassification_3(PredictClassificationBase):
    def execute(self):
        input_base_path = "local/predict_preprocessed"
        output_base_path = "local/predict_3"

        self.train(input_base_path, output_base_path)

    def preprocess(self, df_data_train, df_data_test, df_target_train, df_target_test):
        x_train = df_data_train.values
        x_test = df_data_test.values
        y_train = df_target_train["profit_flag"].values
        y_test = df_target_test["profit_flag"].values

        return x_train, x_test, y_train, y_test

    def model_fit(self, x_train, y_train):
        return ensemble.RandomForestClassifier(n_estimators=200).fit(x_train, y_train)
