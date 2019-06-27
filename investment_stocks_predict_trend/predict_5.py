import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC


def x_y_split(df_prices_preprocessed):
    x = df_prices_preprocessed.drop("date", axis=1).drop("profit_rate", axis=1).drop("profit_flag", axis=1).values
    y = df_prices_preprocessed["profit_flag"].values

    return x, y


def train():
    input_base_path = "local/predict_preprocessed"
    output_base_path = "local/predict_5"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_result = pd.DataFrame()

    for ticker_symbol in df_companies.index:
        print(f"ticker_symbol={ticker_symbol}")

        try:
            df_input = pd.read_csv(f"{input_base_path}/input.{ticker_symbol}.csv", index_col=0)

            df_train = df_input.query("'2008-01-01' <= date <= '2017-12-31'")
            df_test = df_input.query("'2018-01-01' <= date <= '2018-12-31'")
            print(f"df_input len={len(df_input)}")
            print(f"df_train len={len(df_train)}")
            print(f"df_test len={len(df_test)}")

            x_train, y_train = x_y_split(df_train)
            x_test, y_test = x_y_split(df_test)

            clf_best = model_fit(x_train, y_train)
            # joblib.dump(clf_best, f"local/predict_3/random_forest_classifier.{ticker_symbol}.joblib", compress=9)

            # clf = joblib.load(f"local/predict_3/random_forest_classifier.{ticker_symbol}.joblib")
            clf = clf_best
            df_result.at[ticker_symbol, "params"] = clf.get_params().__str__()

            model_score(clf, x_test, y_test, df_result, ticker_symbol)
        except Exception as err:
            print(err)
            df_result.at[ticker_symbol, "error"] = err.__str__()

        print(df_result.loc[ticker_symbol])
        df_result.to_csv(f"{output_base_path}/result.csv")


def model_fit(x_train, y_train, experiment=None):
    return SVC().fit(x_train, y_train)

    parameters = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "gamma": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }

    if experiment is not None:
        experiment.log_parameters(parameters)

    clf = model_selection.GridSearchCV(
        SVC(),
        parameters,
        n_jobs=-1,
        cv=5)

    clf.fit(x_train, y_train)

    best_params = clf.best_params_

    if experiment is not None:
        experiment.log_metrics(best_params)

    clf_best = clf.best_estimator_

    return clf_best


def model_score(clf, x, y, df_result, result_id):
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
