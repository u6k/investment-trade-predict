import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def execute(experiment):
    df_learn, df_test = preprocessing()
    learn_x, learn_y = preprocess_learn_data(df_learn)
    test_x, test_y = preprocess_test_data(df_test)

    model = build_model(learn_x, learn_y)
    model_predict(model, test_x, test_y, experiment)


def preprocessing():
    df_csv = pd.read_csv("local/nikkei_averages.csv")

    df = df_csv.copy()

    df = df[["date", "opening_price", "high_price", "low_price", "close_price"]]
    df = df.sort_values("date")
    df = df.drop_duplicates()
    df = df.assign(id=np.arange(len(df)))
    df = df.set_index("id")

    df_input = df[-600:].copy()
    df_input = df_input.assign(scaled_close_price=sp.minmax_scale(df_input["close_price"]))

    df_learn = df_input[:500].copy()
    df_test = df_input[500:].copy()

    return df_learn, df_test


def preprocess_learn_data(df_learn):
    x, y = [], []

    INPUT_LEN = 20

    for row in range(len(df_learn) - INPUT_LEN):
        x.append(df_learn["scaled_close_price"][row:row+INPUT_LEN].values)
        y.append(df_learn["scaled_close_price"][row+INPUT_LEN:row+INPUT_LEN+1].values)

    x = np.array(x).reshape(len(x), INPUT_LEN, 1)
    y = np.array(y).reshape(len(y), 1)

    print("*** x ***")
    print(len(x))
    print(x)
    print("*** y ***")
    print(len(y))
    print(y)

    return x, y


def preprocess_test_data(df_test):
    test_x, test_y = [], []

    INPUT_LEN = 20

    for row in range(len(df_test) - INPUT_LEN):
        test_x.append(df_test["scaled_close_price"][row:row+INPUT_LEN].values)
        test_y.append(df_test["scaled_close_price"][row+INPUT_LEN:row+INPUT_LEN+1].values)

    test_x = np.array(test_x).reshape(len(test_x), INPUT_LEN, 1)
    test_y = np.array(test_y).reshape(len(test_y), 1)

    print("*** test_x ***")
    print(len(test_x))
    print(test_x)
    print("*** test_y ***")
    print(len(test_y))
    print(test_y)

    return test_x, test_y


def build_model(x, y):
    in_out_neurons = 1
    n_hidden = 300

    batch_size = 128
    epochs = 500

    model = Sequential()
    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(n_hidden))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))

    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    return model


def model_predict(model, test_x, test_y, experiment=None):
    result_y = model.predict(test_x)

    print(len(result_y))
    print(result_y)

    df_result = pd.DataFrame({"id": np.arange(len(test_y)),
                              "original": np.array(test_y).reshape(len(test_y)),
                              "predict": np.array(result_y).reshape(len(result_y))})
    df_result = df_result.set_index("id")

    if experiment is not None:
        experiment.log_asset_data(df_result.to_csv(), file_name="result.csv")

    df_result

    fig = plt.figure(figsize=(20, 5))
    subplot = fig.add_subplot(111)
    subplot.plot(df_result["original"], label="original")
    subplot.plot(df_result["predict"], label="predict")
    subplot.legend()

    plt.show()

    if experiment is not None:
        experiment.log_figure(figure_name="result", figure=fig)
