import numpy as np

from predict_base import PredictClassificationBase


class PredictClassification_7(PredictClassificationBase):
    def train(self):
        pass

    def model_predict(self, ticker_symbol, df_data):
        df_data["predict"] = np.random.randint(0, 2, (len(df_data), ))

        return df_data
