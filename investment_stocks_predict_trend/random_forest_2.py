import pandas as pd
import numpy as np
from sklearn import ensemble, metrics, model_selection
import sklearn.preprocessing as sp



def score(ticker_symbol):
    df_csv = pd.read_csv("local/stock_prices." + str(ticker_symbol) + ".csv")

    df = df_csv.copy()
    df = df[["date", "opening_price", "close_price", "turnover"]]
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

    # scale opening/close price, turnover
    prices = df["opening_price"].values
    prices = np.append(prices, df["close_price"].values)
    prices = np.array(prices).reshape(len(prices), 1)

    scaler = sp.MinMaxScaler()
    scaler.fit(prices)

    prices = df["opening_price"].values
    prices = np.array(prices).reshape(len(prices), 1)
    prices = scaler.transform(prices)
    df = df.assign(scaled_opening_price=prices)

    prices = df["close_price"].values
    prices = np.array(prices).reshape(len(prices), 1)
    prices = scaler.transform(prices)
    df = df.assign(scaled_close_price=prices)

    turnover = df["turnover"].values
    turnover = sp.minmax_scale(turnover)
    df = df.assign(scaled_turnover=turnover)

    # parameters
    INPUT_LEN = 5

    df_train = df[-1500-INPUT_LEN-1:-500].copy()

    df_test = df[-250-INPUT_LEN-1:].copy()

    x_train, y_train = [], []
    for index in df_train.index[:-INPUT_LEN-1]:
        x = []
        for i in range(INPUT_LEN):
            x.append(df_train.at[index+i, "scaled_opening_price"])
            x.append(df_train.at[index+i, "scaled_close_price"])
            x.append(df_train.at[index+i, "scaled_turnover"])
        x_train.append(x)
        y_train.append(int(df_train.at[index+INPUT_LEN, "updown"]))

    x_test, y_test = [], []
    for index in df_test.index[:-INPUT_LEN-1]:
        x = []
        for i in range(INPUT_LEN):
            x.append(df_test.at[index+i, "scaled_opening_price"])
            x.append(df_test.at[index+i, "scaled_close_price"])
            x.append(df_test.at[index+i, "scaled_turnover"])
        x_test.append(x)
        y_test.append(int(df_test.at[index+INPUT_LEN, "updown"]))

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
    df_top_companies = pd.read_csv("local/top_companies.csv", index_col=0)

    df_score = df_top_companies.copy()

    for ticker_symbol in df_score.index:
        ac_score = score(ticker_symbol)

        df_score.at[ticker_symbol, "accuracy_score"] = ac_score
        print(df_score.loc[ticker_symbol])

        df_score.to_csv("local/ac_scores.2.csv")
