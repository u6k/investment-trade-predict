from sklearn import ensemble
from predict_base import PredictClassificationBase


class PredictClassification_3(PredictClassificationBase):
    def execute(self):
        input_base_path = "local/preprocess_4"
        output_base_path = "local/predict_3_preprocess_4"

        self.train(input_base_path, output_base_path)

    def preprocess(self, df_data_train, df_data_test, df_target_train, df_target_test):
        x_train = df_data_train.values
        x_test = df_data_test.values
        y_train = df_target_train["predict_target_label"].values
        y_test = df_target_test["predict_target_label"].values

        return x_train, x_test, y_train, y_test

    def model_fit(self, x_train, y_train):
        return ensemble.RandomForestClassifier(n_estimators=200).fit(x_train, y_train)


if __name__ == "__main__":
    PredictClassification_3().execute()
