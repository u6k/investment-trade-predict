import joblib
import numpy as np
import pandas as pd
from sklearn import ensemble, model_selection


def execute():
    input_base_path = "local/predict_preprocessed"
    output_base_path = "local/predict_3"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)

    df_result = pd.DataFrame(columns=df_companies.columns)

    for ticker_symbol in df_companies.query("message.isnull()").index:
        print(f"ticker_symbol={ticker_symbol}")

        df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        try:
            df_prices_train_data = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.train_data.csv", index_col=0)
            df_prices_train_target = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.train_target.csv", index_col=0)
            df_prices_test_data = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.test_data.csv", index_col=0)
            df_prices_test_target = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.test_target.csv", index_col=0)

            x_train = df_prices_train_data.values
            y_train = df_prices_train_target["profit_flag"].values
            x_test = df_prices_test_data.values
            y_test = df_prices_test_target["profit_flag"].values

            clf = model_fit(x_train, y_train)
            joblib.dump(clf, f"{output_base_path}/random_forest_classifier.{ticker_symbol}.joblib", compress=9)

            model_score(clf, x_test, y_test, df_result, ticker_symbol)
        except Exception as err:
            print(err)
            df_result.at[ticker_symbol, "message"] = err.__str__()

        print(df_result.loc[ticker_symbol])
        df_result.to_csv(f"{output_base_path}/result.csv")


def model_fit(x_train, y_train, experiment=None):
    return ensemble.RandomForestClassifier(n_estimators=200).fit(x_train, y_train)

    parameters = {
        "n_estimators": [10, 100, 200, 500, 1000],
        "max_features": [1, "auto", None],
        "max_depth": [1, 5, 10, 20, 50, None]
    }
    parameters = {
        "n_estimators": [200],
    }

    if experiment is not None:
        experiment.log_parameters(parameters)

    clf = model_selection.GridSearchCV(ensemble.RandomForestClassifier(),
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
