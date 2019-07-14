from sklearn.svm import SVC
from predict_base import PredictClassificationBase


class PredictClassification_5(PredictClassificationBase):
    def execute(self):
        s3_bucket = "u6k"
        input_base_path = "ml-data/stocks/preprocess_5.test"
        output_base_path = "ml-data/stocks/predict_5_preprocess_5.test"

        self.train(s3_bucket, input_base_path, output_base_path)

    def preprocess(self, df_data_train, df_data_test, df_target_train, df_target_test):
        x_train = df_data_train.values
        x_test = df_data_test.values
        y_train = df_target_train["predict_target_label"].values
        y_test = df_target_test["predict_target_label"].values

        return x_train, x_test, y_train, y_test

    def model_fit(self, x_train, y_train):
        return SVC(gamma="scale").fit(x_train, y_train)


if __name__ == "__main__":
    PredictClassification_5().execute()
