import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import Lasso
import sklearn.preprocessing as sp
#import joblib


def preprocess():
    input_base_path = "local/stock_prices_preprocessed"
    output_base_path = "local/predict_4"

    df_companies = pd.read_csv(f"{input_base_path}/companies.csv", index_col=0)
    df_companies = df_companies.query("data_size > 2500")
    df_companies.to_csv(f"{output_base_path}/companies.csv")

    for ticker_symbol in df_companies.index:
        print(ticker_symbol)

        try:
            df_prices_preprocessed = pd.read_csv(f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
            df_prices_simulate_trade_2 = pd.read_csv(f"local/simulate_trade_2/result.{ticker_symbol}.csv", index_col=0)

            df_prices = pd.DataFrame()
            df_prices["id"] = df_prices_preprocessed.index
            df_prices = df_prices.set_index("id")
            df_prices["volume_change"] = sp.minmax_scale(df_prices_preprocessed["volume"].pct_change())
            df_prices["adjusted_close_price_change"] = sp.minmax_scale(df_prices_preprocessed["adjusted_close_price"].pct_change())
            df_prices["sma_5_change"] = sp.minmax_scale(df_prices_preprocessed["sma_5"].pct_change())
            df_prices["sma_10_change"] = sp.minmax_scale(df_prices_preprocessed["sma_10"].pct_change())
            df_prices["sma_20_change"] = sp.minmax_scale(df_prices_preprocessed["sma_20"].pct_change())
            df_prices["sma_40_change"] = sp.minmax_scale(df_prices_preprocessed["sma_40"].pct_change())
            df_prices["sma_80_change"] = sp.minmax_scale(df_prices_preprocessed["sma_80"].pct_change())
            df_prices["profit_rate"] = df_prices_simulate_trade_2["profit_rate"]

            df_prices = df_prices.dropna()
            df_prices.to_csv(f"{output_base_path}/input.{ticker_symbol}.csv")
        except Exception as err:
            print(err)


def x_y_split(df_prices_preprocessed):
    x = df_prices_preprocessed.drop("profit_rate", axis=1).drop("profit_flag", axis=1).values
    y = df_prices_preprocessed["profit_flag"].values

    return x, y


def train():
    base_path = "local/predict_4"

    df_companies = pd.read_csv(f"{base_path}/companies.csv", index_col=0)
    df_result = pd.DataFrame()

    for ticker_symbol in df_companies.index:
        print(f"ticker_symbol={ticker_symbol}")

        try:
            df_input = pd.read_csv(f"{base_path}/input.{ticker_symbol}.csv", index_col=0)

            df_train = df_input[:int(len(df_input)/4*3)]
            df_test = df_input[len(df_train):]
            print(f"df_input len={len(df_input)}")
            print(f"df_train len={len(df_train)}")
            print(f"df_test len={len(df_test)}")

            x_train, y_train = x_y_split(df_train)
            x_test, y_test = x_y_split(df_test)

            clf_best = model_fit(x_train, y_train)
            # joblib.dump(clf_best, f"{base_path}/lasso.{ticker_symbol}.joblib", compress=9)

            # clf = joblib.load(f"{base_path}/lasso.{ticker_symbol}.joblib")
            clf = clf_best
            df_result.at[ticker_symbol, "params"] = clf.get_params().__str__()

            score = clf.score(x_test, y_test)
            print(f"score={score}")
            df_result.at[ticker_symbol, "score"] = score

            df_test_2 = df_test.query("profit_rate>1.0")
            print(f"df_test_2 len={len(df_test_2)}")

            x_test_2, y_test_2 = x_y_split(df_test_2)

            score_2 = clf.score(x_test_2, y_test_2)
            print(f"score_2={score_2}")
            df_result.at[ticker_symbol, "score_2"] = score_2

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
                                       n_jobs=-1)

    clf.fit(x_train, y_train)

    best_params = clf.best_params_

    if experiment is not None:
        experiment.log_metrics(best_params)

    clf_best = clf.best_estimator_

    return clf_best
