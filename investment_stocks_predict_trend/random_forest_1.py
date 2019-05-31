import pandas as pd
import numpy as np
from sklearn import ensemble, metrics, model_selection
import sklearn.preprocessing as sp


def score(ticker_symbol):
    df_csv = pd.read_csv("local/stock_prices." + str(ticker_symbol) + ".csv")

    df = df_csv.copy()
    df = df[["date", "opening_price", "close_price"]]
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

    # parameters
    x, y = [], []

    for i in range(1, 1001):
        x.append(df[-i-31:-i-1]["return_index"].values)
        y.append(int(df[-i-1:-i]["updown"].values))

    x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(x, y)

    # build model
    params = {
        "n_estimators": [100, 500, 1000],
        "criterion": ["gini", "entropy"],
        "max_depth": [4, 16, 64],
        "class_weight": ["balanced"]
    }

    clf = model_selection.GridSearchCV(
        ensemble.RandomForestClassifier(),
        params,
        n_jobs=-1,
        verbose=10
    )

    clf.fit(x_train, y_train)

    best_params = clf.best_params_

    print("*** best params ***")
    print(best_params)

    clf_best = clf.best_estimator_

    result = clf_best.predict(x_test)
    ac_score = metrics.accuracy_score(y_test, result)

    print("*** accuracy score ***")
    print(ac_score)

    return ac_score

def scores():
    df_score = pd.read_csv("local/top_companies.csv", index_col=0)

    for ticker_symbol in df_score.index:
        ac_score = score(ticker_symbol)

        df_score.at[ticker_symbol, "accuracy_score"] = ac_score
        print(df_score.loc[ticker_symbol])

        df_score.to_csv("local/ac_scores.csv")
