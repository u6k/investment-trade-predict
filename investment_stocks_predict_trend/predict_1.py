import pandas as pd
import numpy as np
from sklearn import ensemble, metrics, model_selection


def execute(experiment):
    x_train, x_test, y_train, y_test = preprocessing()
    clf_best = model_fit(x_train, y_train, experiment)
    model_score(clf_best, x_test, y_test, experiment)


def preprocessing():
    df_csv = pd.read_csv("drive/My Drive/projects/ml_data/stocks/nikkei_averages.csv", index_col=0)
    df_csv

    df = df_csv.copy()

    df = df[["date", "opening_price", "high_price", "low_price", "close_price"]]
    df = df.sort_values("date")
    df = df.drop_duplicates()
    df = df.assign(id=np.arange(len(df)))
    df = df.set_index("id")

    # updown
    updown = [np.nan]
    for id in df[:-1].index:
        if df.at[id+1, "close_price"] > df.at[id, "close_price"]:
            updown.append(1)
        else:
            updown.append(0)

    df = df.assign(updown=updown)

    # return index
    returns = df["close_price"].pct_change()
    return_index = (1 + returns).cumprod()

    df = df.assign(return_index=return_index)

    x, y = [], []

    for i in range(1, 1001):
        x.append(df[-i-31:-i-1]["return_index"].values)
        y.append(int(df[-i-1:-i]["updown"].values))

    print(x)
    print(y)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

    print("x_train.len:", len(x_train))
    print("x_test.len:", len(x_test))
    print("y_train.len:", len(y_train))
    print("y_test.len:", len(y_test))

    return x_train, x_test, y_train, y_test


def model_fit(x_train, y_train, experiment=None):
    parameters = {
        "n_estimators": [10, 100, 200],
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 4, 8, 16, 32, 64],
        "random_state": [1, 2, 3],
        "class_weight": ["balanced"]
    }

    if experiment is not None:
        experiment.log_parameters(parameters)

    clf = model_selection.GridSearchCV(ensemble.RandomForestClassifier(),
                                       parameters,
                                       cv=5,
                                       n_jobs=-1,
                                       verbose=3)

    clf.fit(x_train, y_train)

    best_params = clf.best_params_

    if experiment is not None:
        experiment.log_metrics(best_params)

    print("*** best params ***")
    print(best_params)

    clf_best = clf.best_estimator_

    return clf_best


def model_score(clf_best, x_test, y_test, experiment=None):
    result = clf_best.predict(x_test)
    ac_score = metrics.accuracy_score(y_test, result)

    if experiment is not None:
        experiment.log_metric("accuracy_score", ac_score)

    print("*** accuracy score ***")
    print(ac_score)
