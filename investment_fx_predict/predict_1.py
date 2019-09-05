import argparse
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import app_s3
from app_logging import get_app_logger


def execute(fxpairs, train_start_date, train_end_date, test_start_date, test_end_date, input_explanatory_base_path, input_explained_base_path, output_base_path):
    L = get_app_logger()
    L.info(f"start: fxpairs={fxpairs}, train_start_date={train_start_date}, train_end_date={train_end_date}, test_start_date={test_start_date}, test_end_date={test_end_date}")

    # Load index
    df_fxpairs = app_s3.read_dataframe(
        f"{input_explanatory_base_path}/fxpairs.csv",
        dtype={"id": int, "fxpair": str, "start_date": str, "end_date": str},
        parse_dates=["start_date", "end_date"],
        index_col=0
    )

    # Train
    df_report = pd.DataFrame(columns=["fxpair", "start_date", "end_date", "true_positive", "false_positive", "false_negative", "true_negative"])

    for fxpair in fxpairs:
        # Preprocess
        x_train, y_train = None, None

        for id in df_fxpairs.query(f"fxpair=='{fxpair}' and '{train_start_date.strftime('%Y-%m-%d')}'<=start_date<'{train_end_date.strftime('%Y-%m-%d')}'").index:
            x_train_sub, y_train_sub = preprocess_train(fxpair, df_fxpairs.at[id, "start_date"], df_fxpairs.at[id, "end_date"], input_explanatory_base_path, input_explained_base_path)

            if x_train is None:
                x_train = x_train_sub
                y_train = y_train_sub
            else:
                x_train = np.concatenate([x_train, x_train_sub])
                y_train = np.concatenate([y_train, y_train_sub])

            L.info(f"x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")

        # Train model
        model = train_model(x_train, y_train)

        # Save model
        app_s3.write_sklearn_model(model, f"{output_base_path}/model.{fxpair}.{train_start_date.strftime('%Y%m%d')}_{train_end_date.strftime('%Y%m%d')}.joblib")

        # Report model
        for id in df_fxpairs.query(f"fxpair=='{fxpair}' and '{test_start_date.strftime('%Y-%m-%d')}'<=start_date<'{test_end_date.strftime('%Y-%m-%d')}'").index:
            x_test, y_test = preprocess_train(fxpair, df_fxpairs.at[id, "start_date"], df_fxpairs.at[id, "end_date"], input_explanatory_base_path, input_explained_base_path, False)

            result = report_model(fxpair, df_fxpairs.at[id, "start_date"], df_fxpairs.at[id, "end_date"], x_test, y_test, model)

            report_id = len(df_report)
            df_report.at[report_id, "fxpair"] = result["fxpair"]
            df_report.at[report_id, "start_date"] = result["start_date"]
            df_report.at[report_id, "end_date"] = result["end_date"]
            df_report.at[report_id, "false_negative"] = result["false_negative"]
            df_report.at[report_id, "true_negative"] = result["true_negative"]

            app_s3.write_dataframe(df_report, f"{output_base_path}/report.csv")

    L.info("finish")


def preprocess_train(fxpair, start_date, end_date, input_explanatory_base_path, input_explained_base_path, resampling=True):
    L = get_app_logger()
    L.info(f"preprocess_train: fxpair={fxpair}, start_date={start_date}, end_date={end_date}")

    # Load data
    df_explanatory = app_s3.read_dataframe(f"{input_explanatory_base_path}/fx.{fxpair}.{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv", index_col=0)
    df_explained = app_s3.read_dataframe(f"{input_explained_base_path}/fx.{fxpair}.{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv", index_col=0)

    # Preprocess
    df = df_explanatory.drop("timestamp", axis=1)
    df["predict_target_label"] = df_explained["profit_rate"].apply(lambda r: 1 if r > 0.001 else 0)

    df = df.dropna()

    # Under sampling
    x = df.drop("predict_target_label", axis=1).values
    y = df["predict_target_label"].values

    L.info(f"Load data: x.shape={x.shape}, y.shape={y.shape}")

    if resampling:
        x, y = RandomUnderSampler(random_state=0).fit_sample(x, y)

        L.info(f"Under sampled: x.shape={x.shape}, y.shape={y.shape}")

    return x, y


def train_model(x_train, y_train):
    L = get_app_logger()
    L.info("train_model")

    params = {
        "C": [0.000001, 0.00001, 0.0001],
        "kernel": ["rbf"],
        "gamma": ["scale"],
        "random_state": [0],
        "class_weight": ["balanced"]
    }

    clf = model_selection.GridSearchCV(
        SVC(),
        params,
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    clf.fit(x_train, y_train)

    L.info(f"best_params: {clf.best_params_}")

    clf_best = clf.best_estimator_

    return clf_best


def report_model(fxpair, start_date, end_date, x_test, y_test, model):
    L = get_app_logger()
    L.info(f"report_model: start_date={start_date}, end_date={end_date}")

    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)

    result = {
        "fxpair": fxpair,
        "start_date": start_date,
        "end_date": end_date,
        "true_positive": cm[0][0],
        "false_positive": cm[0][1],
        "false_negative": cm[1][0],
        "true_negative": cm[1][1],
    }

    L.info(result)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    execute(
        fxpairs=["usdjpy"],
        train_start_date=datetime(2018, 1, 1, 0, 0, 0),
        train_end_date=datetime(2018, 2, 1, 0, 0, 0),
        test_start_date=datetime(2018, 1, 1, 0, 0, 0),
        test_end_date=datetime(2018, 3, 1, 0, 0, 0),
        input_explanatory_base_path="preprocess_3",
        input_explained_base_path="preprocess_4",
        output_base_path="predict_1"
    )
