import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import Lasso


def x_y_split(df_prices_preprocessed):
    x = df_prices_preprocessed.drop("date", axis=1).drop("profit_rate", axis=1).drop("profit_flag", axis=1).values
    y = df_prices_preprocessed["profit_rate"].values

    return x, y


def train():
    base_path = "local/predict_4"

    df_companies = pd.read_csv(f"{base_path}/companies.csv", index_col=0)
    df_result = pd.DataFrame()

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

            ac_score = model_score(clf, x_test, y_test)
            print(f"ac_score={ac_score}")
            df_result.at[ticker_symbol, "ac_score"] = ac_score

            df_test_2 = df_test.query("profit_rate>1.0")
            print(f"df_test_2 len={len(df_test_2)}")

            x_test_2, y_test_2 = x_y_split(df_test_2)

            ac_score_2 = model_score(clf, x_test_2, y_test_2)
            print(f"ac_score_2={ac_score_2}")
            df_result.at[ticker_symbol, "ac_score_2"] = ac_score_2

            df_result.to_csv(f"{base_path}/result.csv")
        except Exception as err:
            print(err)


def model_fit(x_train, y_train, experiment=None):
    parameters = {
        "alpha": [0.1, 0.5, 1.0, 10.0, 100.0]
    }

    if experiment is not None:
        experiment.log_parameters(parameters)

    clf = model_selection.GridSearchCV(Lasso(),
                                       parameters,
                                       n_jobs=-1,
                                       cv=5)

    clf.fit(x_train, y_train)

    best_params = clf.best_params_

    if experiment is not None:
        experiment.log_metrics(best_params)

    clf_best = clf.best_estimator_

    return clf_best


def model_score(clf, x, y):
    y_pred = clf.predict(x)

    count = 0.0

    for i in range(len(y)):
        if (y[i] >= 1.0 and y_pred[i] >= 1.0) or (y[i] < 1.0 and y_pred[i] < 1.0):
            count += 1.0

    ac_score = count / len(y)

    return ac_score
