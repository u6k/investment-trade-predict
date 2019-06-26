import numpy as np
import pandas as pd
from sklearn import ensemble, metrics, model_selection
# import joblib


def preprocess():
    input_base_path = "local/stock_prices_preprocessed"
    output_base_path = "local/predict_3"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies = df_companies.query("data_size > 2500")
    df_companies.to_csv(f"{output_base_path}/companies.csv")

    for ticker_symbol in df_companies.index:
        print(ticker_symbol)

        df_prices_preprocessed = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
        df_prices_simulate_trade_2 = pd.read_csv(f"local/simulate_trade_2/result.{ticker_symbol}.csv", index_col=0)

        df_prices = pd.DataFrame()
        df_prices["id"] = df_prices_preprocessed.index
        df_prices = df_prices.set_index("id")
        df_prices["date"] = df_prices_preprocessed["date"]
        df_prices["volume_change"] = df_prices_preprocessed["volume"] / df_prices_preprocessed["volume"].shift(1)
        df_prices["adjusted_close_price_change"] = df_prices_preprocessed["adjusted_close_price"] / df_prices_preprocessed["adjusted_close_price"].shift(1)
        df_prices["sma_5_change"] = df_prices_preprocessed["sma_5"] / df_prices_preprocessed["sma_5"].shift(1)
        df_prices["sma_10_change"] = df_prices_preprocessed["sma_10"] / df_prices_preprocessed["sma_10"].shift(1)
        df_prices["sma_20_change"] = df_prices_preprocessed["sma_20"] / df_prices_preprocessed["sma_20"].shift(1)
        df_prices["sma_40_change"] = df_prices_preprocessed["sma_40"] / df_prices_preprocessed["sma_40"].shift(1)
        df_prices["sma_80_change"] = df_prices_preprocessed["sma_80"] / df_prices_preprocessed["sma_80"].shift(1)
        df_prices["profit_rate"] = df_prices_simulate_trade_2["profit_rate"]
        df_prices["profit_flag"] = df_prices_simulate_trade_2["profit_rate"].apply(lambda r: 1 if r > 1.0 else 0)

        df_prices = df_prices.dropna()
        df_prices.to_csv(f"local/predict_3/input.{ticker_symbol}.csv")


def x_y_split(df_prices_preprocessed):
    x = df_prices_preprocessed.drop("date", axis=1).drop("profit_rate", axis=1).drop("profit_flag", axis=1).values
    y = df_prices_preprocessed["profit_flag"].values

    return x, y


def train():
    base_path = "local/predict_3"

    df_companies = pd.read_csv(f"{base_path}/companies.csv", index_col=0)
    df_result = df_companies[["name", "data_size"]].copy()

    for ticker_symbol in df_companies.index:
        print(f"ticker_symbol={ticker_symbol}")

        try:
            df_input = pd.read_csv(f"{base_path}/input.{ticker_symbol}.csv", index_col=0)

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
        df_result.to_csv(f"{base_path}/result.csv")


def model_fit(x_train, y_train, experiment=None):
    return ensemble.RandomForestClassifier(n_estimators=200).fit(x_train, y_train)

    # parameters = {
    #    "n_estimators": [10, 100, 200, 500, 1000],
    #    "max_features": [1, "auto", None],
    #    "max_depth": [1, 5, 10, 20, 50, None]
    # }
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
