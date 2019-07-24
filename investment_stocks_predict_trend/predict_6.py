import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping

from app_logging import get_app_logger
import app_s3
from predict_base import PredictClassificationBase


class PredictClassification_6(PredictClassificationBase):
    def preprocess_impl(self, ticker_symbol):
        L = get_app_logger(f"preprocess.{ticker_symbol}")
        L.info(f"predict preprocess_6: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        period = 32

        try:
            # Load data
            df_preprocess = app_s3.read_dataframe(self._s3_bucket, f"{self._input_preprocess_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            # Preprocess
            df = df_preprocess[[
                "date",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "adjusted_close_price",
                "volume"
            ]].copy()

            df["profit"] = df["close_price"]-df["open_price"]
            df["profit_rate"] = df["profit"] / df["close_price"]
            df["profit_rate_minmax"] = MinMaxScaler().fit_transform(df["profit_rate"].values.reshape(-1, 1))

            for i in range(0, period):
                df[f"profit_rate_minmax_{i}"] = df["profit_rate_minmax"].shift(i)

            df["profit_rate_target"] = df["profit_rate"].shift(-1)
            df["predict_target"] = df["profit_rate_target"].apply(lambda r: 1 if r > 0.005 else 0)

            df = df.drop(["profit", "profit_rate", "profit_rate_minmax", "profit_rate_target"], axis=1)

            df = df.dropna()

            # Save data
            app_s3.write_dataframe(df, self._s3_bucket, f"{self._output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def train(self):
        L = get_app_logger("train_6")
        L.info("start")

        df_companies = app_s3.read_dataframe(self._s3_bucket, f"{self._input_preprocess_base_path}/companies.csv", index_col=0)
        df_result = pd.DataFrame(columns=df_companies.columns)

        for ticker_symbol in df_companies.index:
            result = self.train_impl(ticker_symbol)

            if result["exception"] is not None:
                continue

            ticker_symbol = result["ticker_symbol"]
            scores = result["scores"]

            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]
            for key in scores.keys():
                df_result.at[ticker_symbol, key] = scores[key]

        app_s3.write_dataframe(df_result, self._s3_bucket, f"{self._output_base_path}/report.csv")

        L.info("finish")

    def train_impl(self, ticker_symbol):
        L = get_app_logger(ticker_symbol)
        L.info(f"train_6: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None,
            "scores": None
        }

        try:
            x_train, x_test, y_train, y_test = self.train_test_split(ticker_symbol)

            model = self.model_fit(x_train, y_train)
            app_s3.write_keras_model(model, self._s3_bucket, f"{self._output_base_path}/model.{ticker_symbol}")

            result["scores"] = self.model_score(model, x_test, y_test)
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def model_fit(self, x_train, y_train):
        x = x_train.reshape(len(x_train), len(x_train[0]), 1)
        y = keras.utils.to_categorical(y_train, 2)

        model = Sequential()
        model.add(Conv1D(64, kernel_size=8, input_shape=(32, 1), activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(2, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adadelta(),
            metrics=["accuracy"]
        )

        model.fit(x, y, batch_size=128, epochs=100, verbose=0, validation_split=0.2, callbacks=[EarlyStopping(patience=10)])

        return model

    def model_score(self, model, x_test, y_test):
        x = x_test.reshape(len(x_test), len(x_test[0]), 1)
        y = keras.utils.to_categorical(y_test, 2)

        score = model.evaluate(x, y, batch_size=100)

        return {
            "loss": score[0],
            "accuracy": score[1]
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="preprocess, or train")
    parser.add_argument("--suffix", help="folder name suffix (default: test)", default="test")
    args = parser.parse_args()

    pred = PredictClassification_6(
        train_start_date="2008-01-01",
        train_end_date="2017-12-31",
        test_start_date="2018-01-01",
        test_end_date="2018-12-31",
        s3_bucket="u6k",
        input_preprocess_base_path=f"ml-data/stocks/preprocess_1.{args.suffix}",
        input_simulate_base_path=None,
        output_base_path=f"ml-data/stocks/predict_6.{args.suffix}"
    )

    if args.task == "preprocess":
        pred.preprocess()
    elif args.task == "train":
        pred.train()
    else:
        parser.print_help()
