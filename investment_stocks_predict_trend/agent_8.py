import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import chainer
import chainerrl


def execute(experiment):
    df = preprocessing()
    env = build_env(df)
    agent = build_agent(env, experiment)
    learn_agent(env, agent, experiment)
    df_result = simulate_agent(env, agent, experiment)
    build_figure_win_vs_lose(df_result, experiment)
    build_figure_reward(df_result, experiment)


def preprocessing():
    df_csv = pd.read_csv("local/stock_prices/stock_prices.7974.csv")

    df = df_csv.copy()

    df = df.query("ticker_symbol == '7974'").copy()
    df = df[["date", "opening_price", "high_price", "low_price", "close_price", "turnover", "adjustment_value"]]
    df = df.sort_values("date")
    df = df.drop_duplicates()
    df = df.assign(id=np.arange(len(df)))
    df = df.set_index("id")

    df_input = df[-600:].copy()

    price = df_input["opening_price"].values
    price = np.append(price, df_input["close_price"].values)
    price = np.array(price).reshape(len(price), 1)

    scaler = sp.MinMaxScaler()
    scaler.fit(price)

    opening_price = df_input["opening_price"].values
    opening_price = np.array(opening_price).reshape(len(opening_price), 1)
    df_input["scaled_opening_price"] = scaler.transform(opening_price)

    close_price = df_input["close_price"].values
    close_price = np.array(close_price).reshape(len(close_price), 1)
    df_input["scaled_close_price"] = scaler.transform(close_price)

    df_input

    open_x, open_y, close_x, close_y = [], [], [], []

    INPUT_LEN = 20

    for row in range(len(df_input) - INPUT_LEN):
        open_x.append(df_input["scaled_opening_price"][row:row+INPUT_LEN].values)
        open_y.append(df_input["scaled_opening_price"][row+INPUT_LEN:row+INPUT_LEN+1].values)
        close_x.append(df_input["scaled_close_price"][row:row+INPUT_LEN].values)
        close_y.append(df_input["scaled_close_price"][row+INPUT_LEN:row+INPUT_LEN+1].values)

    open_x = np.array(open_x).reshape(len(open_x), INPUT_LEN, 1)
    open_y = np.array(open_y).reshape(len(open_y), 1)
    close_x = np.array(close_x).reshape(len(close_x), INPUT_LEN, 1)
    close_y = np.array(close_y).reshape(len(close_y), 1)

    print("*** open_x ***")
    print(open_x)
    print("*** open_y ***")
    print(open_y)
    print("*** close_x ***")
    print(close_x)
    print("*** close_y ***")
    print(close_y)

    length_of_sequence = INPUT_LEN
    in_out_neurons = 1
    n_hidden = 300

    batch_size = 128
    epochs = 500

    open_model = Sequential()
    open_model.add(LSTM(n_hidden,
                        batch_input_shape=(None, length_of_sequence, in_out_neurons),
                        return_sequences=False))
    open_model.add(Dense(in_out_neurons))
    open_model.add(Activation("linear"))
    open_model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))

    open_history = open_model.fit(open_x,
                                  open_y,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_split=0.2,
                                  callbacks=[EarlyStopping(patience=10, verbose=1)])

    close_model = Sequential()
    close_model.add(LSTM(n_hidden,
                         batch_input_shape=(None, length_of_sequence, in_out_neurons),
                         return_sequences=False))
    close_model.add(Dense(in_out_neurons))
    close_model.add(Activation("linear"))
    close_model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))

    close_history = close_model.fit(close_x,
                                    close_y,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_split=0.2,
                                    callbacks=[EarlyStopping(patience=10, verbose=1)])

    loss = open_history.history["loss"]
    val_loss = open_history.history["val_loss"]
    epochs = range(1, len(loss)+1)

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.show()

    loss = close_history.history["loss"]
    val_loss = close_history.history["val_loss"]
    epochs = range(1, len(loss)+1)

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.show()

    for idx in df_input[: -INPUT_LEN].index:
        print(str(idx))
        result_open_price, result_close_price = [], []
        open_price_subset = [df_input.at[idx+i, "scaled_opening_price"] for i in range(INPUT_LEN)]
        close_price_subset = [df_input.at[idx+i, "scaled_close_price"] for i in range(INPUT_LEN)]

        for i in range(INPUT_LEN):
            future = open_price_subset[i: INPUT_LEN]
            future = np.append(future, result_open_price)
            future = np.array(future).reshape(1, INPUT_LEN, 1)

            result = open_model.predict(future)

            result_open_price = np.append(result_open_price, result)

            future = close_price_subset[i: INPUT_LEN]
            future = np.append(future, result_close_price)
            future = np.array(future).reshape(1, INPUT_LEN, 1)

            result = close_model.predict(future)

            result_close_price = np.append(result_close_price, result)

        for i in range(INPUT_LEN):
            df_input.at[idx+INPUT_LEN-1, "predict_opening_price_"+str(i)] = result_open_price[i]
            df_input.at[idx+INPUT_LEN-1, "predict_close_price_"+str(i)] = result_close_price[i]

    df_input

    df_tmp = pd.DataFrame({"input": df_input["scaled_opening_price"], "predict": 0.})
    for i in range(20):
        df_tmp.at[8000+i, "predict"] = df_input.at[7999+i, "predict_opening_price_"+str(i)]

    df_tmp[-50:].plot()

    df_input["scaled_opening_price"].plot()
    df_input["predict_opening_price_1"].plot()

    df_input["scaled_close_price"].plot()
    df_input["predict_close_price_1"].plot()

    return df_input


class LearnEnv():
    def __init__(self, df, start_id, end_id):
        self.DF = df.copy()
        self.START_ID = start_id
        self.END_ID = end_id

        self.reset()

        self.data_len = self.END_ID - self.START_ID
        self.action_size = 2  # 0...何もしない、1...購入or売却
        self.observation_size = len(self.observe())

    def reset(self):
        self.total_reward = 0.0
        self.funds = 0.0
        self.current_id = self.START_ID
        self.buy_price = 0.0
        self.predict_buy_price = 0.0
        self.done = False
        self.win = 0
        self.lose = 0

        self.df_action = self.DF.copy()
        self.df_action = self.df_action.assign(reward=0.)
        self.df_action = self.df_action.assign(funds=0.)
        self.df_action = self.df_action.assign(buy=0)
        self.df_action = self.df_action.assign(sell=0)
        self.df_action = self.df_action.assign(win=0)
        self.df_action = self.df_action.assign(lose=0)

        return self.observe()

    def step(self, action):
        if action == 0:
            reward = 0.0
        elif self.buy_price == 0.0:
            # buy
            self.buy_price = self.df_action.at[self.current_id, "scaled_opening_price"]
            self.predict_buy_price = self.df_action.at[self.current_id-1, "predict_opening_price_1"]
            self.funds -= self.buy_price
            reward = 0.0

            self.df_action.at[self.current_id, "buy"] = 1
        elif self.buy_price != 0.0:
            # sell
            sell_price = self.df_action.at[self.current_id, "scaled_close_price"]
            self.funds += sell_price
            reward = sell_price - self.buy_price
            self.total_reward += reward
            self.buy_price = 0.0
            self.predict_buy_price = 0.0

            if reward > 0:
                self.win += 1
            else:
                self.lose += 1

            self.df_action.at[self.current_id, "sell"] = 1

        self.df_action.at[self.current_id, "reward"] = self.total_reward
        self.df_action.at[self.current_id, "funds"] = self.funds
        self.df_action.at[self.current_id, "win"] = self.win
        self.df_action.at[self.current_id, "lose"] = self.lose

        self.current_id += 1
        if self.current_id >= self.END_ID:
            self.done = True

        return self.observe(), reward, self.done, {}

    def render(self):
        print(self.df_action.loc[self.current_id-1])

    def observe(self):
        obs = np.array(
            [self.df_action.at[self.current_id - i, "opening_price"] for i in range(1, 4)],
            dtype=np.float32
        )
        obs = np.append(obs, np.array(
            [self.df_action.at[self.current_id - i, "predict_opening_price_"+str(i)] for i in range(0, 3)],
            dtype=np.float32
        ))
        obs = np.append(obs, np.array(
            [self.df_action.at[self.current_id - i, "close_price"] for i in range(1, 4)],
            dtype=np.float32
        ))
        obs = np.append(obs, np.array(
            [self.df_action.at[self.current_id - i, "predict_close_price_"+str(i)] for i in range(0, 3)],
            dtype=np.float32
        ))
        obs = np.append(obs, np.array(self.buy_price, dtype=np.float32))
        obs = np.append(obs, np.array(self.predict_buy_price, dtype=np.float32))

        return obs

    def random_action(self):
        return np.random.randint(0, 2)


def build_env(df):
    env = LearnEnv(df, 8036-250, 8036)

    return env


def build_agent(env, experiment=None):
    hyper_params = {
        "n_hidden_layers": 3,
        "obs_size": env.observation_size,
        "n_actions": env.action_size,
        "n_hidden_channels": env.observation_size * env.action_size,
        "adam_eps": 1e-2,
        "gamma": 0.95,
        "start_epsilon": 1.0,
        "end_epsilon": 0.3,
        "decay_steps": 200 * env.data_len,
        "replay_buffer_capacity": 10 ** 6,
        "ddqn_replay_start_size": 500,
        "ddqn_update_interval": 1,
        "ddqn_target_update_interval": 100
    }
    if experiment is not None:
        experiment.log_parameters(hyper_params)

    q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        hyper_params["obs_size"],
        hyper_params["n_actions"],
        n_hidden_layers=hyper_params["n_hidden_layers"],
        n_hidden_channels=hyper_params["n_hidden_channels"]
    )
    # q_func.to_gpu(0)

    optimizer = chainer.optimizers.Adam(eps=hyper_params["adam_eps"])
    optimizer.setup(q_func)

    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=hyper_params["start_epsilon"],
        end_epsilon=hyper_params["end_epsilon"],
        decay_steps=hyper_params["decay_steps"],
        random_action_func=env.random_action
    )

    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=hyper_params["replay_buffer_capacity"])

    agent = chainerrl.agents.DoubleDQN(
        q_func,
        optimizer,
        replay_buffer,
        hyper_params["gamma"],
        explorer,
        replay_start_size=hyper_params["ddqn_replay_start_size"],
        update_interval=hyper_params["ddqn_update_interval"],
        target_update_interval=hyper_params["ddqn_target_update_interval"]
    )

    return agent


def learn_agent(env, agent, experiment=None):
    n_episodes = 500

    for i in range(1, n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        R = 0

        while not done:
            action = agent.act_and_train(obs, reward)
            obs, reward, done, _ = env.step(action)
            R += reward

        agent.stop_episode_and_train(obs, reward, done)

        metrics = {
            "reward": R,
            "epsilon": agent.explorer.epsilon,
            "win": env.win,
            "lose": env.lose,
            "funds": env.funds + env.buy_price
        }
        if experiment is not None:
            experiment.log_metrics(metrics, step=i)

        if i % 10 == 0:
            print("episode:", i, ", R:", R, ", statistics:", agent.get_statistics(), ", epsilon:", agent.explorer.epsilon)
            env.render()


def simulate_agent(env, agent, experiment=None):
    obs = env.reset()
    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)

        env.render()

    agent.stop_episode()

    df_result = env.df_action.query("7786 <= id").copy()

    if experiment is not None:
        experiment.log_asset_data(df_result.to_csv(), file_name="result.csv")

    return df_result


def build_figure_win_vs_lose(df_result, experiment=None):
    fig = plt.figure(figsize=(20, 5))
    subplot = fig.add_subplot(111)
    subplot.plot(df_result["win"], label="win")
    subplot.plot(df_result["lose"], label="lose")
    subplot.legend()

    plt.show()

    if experiment is not None:
        experiment.log_figure(figure_name="win_vs_lose", figure=fig)


def build_figure_reward(df_result, experiment=None):
    fig = plt.figure(figsize=(20, 5))
    subplot = fig.add_subplot(222)
    subplot.plot(df_result["reward"], label="reward")
    subplot.legend()

    plt.show()

    if experiment is not None:
        experiment.log_figure(figure_name="reward", figure=fig)
