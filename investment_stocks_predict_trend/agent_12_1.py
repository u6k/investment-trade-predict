import pandas as pd
import numpy as np
import chainer
import chainerrl
import matplotlib.pyplot as plt


def execute(experiment=None, max_episode=500):
    TICKER_SYMBOL = "5610"

    df = load_data(TICKER_SYMBOL)
    print(df)

    train_env = TrainEnv(df, 5881, 7057)
    test_env = TrainEnv(df, 7057, 7750)

    agent = build_agent(train_env, experiment)

    for i in range(1, max_episode+1):
        print("*** episode: "+str(i)+" ***")
        df_result, metrics = train_agent(train_env, agent)
        if experiment is not None:
            experiment.log_metrics(metrics, step=i)

        if i % 100 == 0:
            print("episode: "+str(i))
            print(metrics)

            df_result, metrics = simulate_agent(test_env, agent)
            if experiment is not None:
                experiment.log_asset_data(df_result.to_csv(), file_name="test_result."+str(i)+".csv")

            build_figure_result(df_result, experiment)


def load_data(ticker_symbol):
    df = pd.read_csv(f"local/stock_prices/stock_prices.{ticker_symbol}.csv")

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.sort_values("date")
    df["id"] = np.arange(len(df))
    df = df.set_index("id")

    return df


class TrainEnv():
    def __init__(self, df, start_id, end_id):
        self.DF = df.copy()
        self.START_ID = start_id
        self.END_ID = end_id

        self.reset()

        self.data_len = self.END_ID - self.START_ID
        self.action_size = 2  # 0...stay, 1...buy or sell
        self.observation_size = len(self.observe())

    def reset(self):
        self.total_reward = 0.0
        self.funds = 1000000
        self.assets = self.funds
        self.buy_price = 0.0
        self.buy_stocks = 0
        self.current_id = self.START_ID
        self.done = False
        self.win = 0
        self.lose = 0

        self.df_result = self.DF.copy()

        return self.observe()

    def step(self, action):
        if action == 0:
            reward = 0.0
        elif self.buy_stocks == 0:
            # buy
            self.buy_price = self.df_result.at[self.current_id, "close_price"]
            self.buy_stocks = (self.funds * 0.5) // (self.buy_price * 100) * 100
            self.funds -= self.buy_price * self.buy_stocks

            reward = 0.0
        else:
            # sell
            sell_price = self.df_result.at[self.current_id, "close_price"]
            reward = sell_price - self.buy_price
            self.total_reward += reward

            self.funds += sell_price * self.buy_stocks
            self.buy_price = 0.0
            self.buy_stocks = 0

            if reward > 0:
                self.win += 1
            else:
                self.lose += 1

        self.assets = self.funds + self.df_result.at[self.current_id, "close_price"] * self.buy_stocks

        self.df_result.at[self.current_id, "total_reward"] = self.total_reward
        self.df_result.at[self.current_id, "funds"] = self.funds
        self.df_result.at[self.current_id, "assets"] = self.assets
        self.df_result.at[self.current_id, "buy_price"] = self.buy_price
        self.df_result.at[self.current_id, "buy_stocks"] = self.buy_stocks
        self.df_result.at[self.current_id, "win"] = self.win
        self.df_result.at[self.current_id, "lose"] = self.lose

        self.current_id += 1
        if self.current_id >= self.END_ID:
            self.done = True

        return self.observe(), reward, self.done, {}

    def render(self):
        print(f"id: {self.current_id}")
        print(self.df_result[self.current_id, self.current_id+1])
        print(f"observe: {self.observe()}")

    def observe(self):
        obs = np.array(
            [self.df_result.at[self.current_id-i, "adjusted_close_price"] for i in range(1, 21)],
            dtype=np.float32
        )
        obs = np.append(obs, np.array(
            [self.df_result.at[self.current_id-i, "volume"] for i in range(1, 21)],
            dtype=np.float32
        ))

        return obs

    def random_action(self):
        return np.random.randint(0, 2)


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
        "decay_steps": env.data_len * 200,
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


def train_agent(env, agent):
    obs = env.reset()
    reward = 0
    done = False

    # env.render()

    while not done:
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)

        # env.render()

    agent.stop_episode_and_train(obs, reward, done)

    metrics = {
        "reward": env.total_reward,
        "epsilon": agent.explorer.epsilon,
        "win": env.win,
        "lose": env.lose,
        "funds": env.funds,
        "assets": env.assets
    }

    df_result = env.df_result.query(f"{env.START_ID} <= id <= {env.END_ID}").copy()

    return df_result, metrics


def simulate_agent(env, agent):
    obs = env.reset()
    done = False

    # env.render()

    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)

        # env.render()

    agent.stop_episode()

    metrics = {
        "reward": env.total_reward,
        "epsilon": agent.explorer.epsilon,
        "win": env.win,
        "lose": env.lose,
        "funds": env.funds,
        "assets": env.assets
    }

    df_result = env.df_result.query(f"{env.START_ID} <= id <= {env.END_ID}").copy()

    return df_result, metrics


def build_figure_result(df_result, experiment=None):
    fig = plt.figure(figsize=(20, 5))
    subplot = fig.add_subplot(111)
    subplot.plot(df_result["win"], label="win")
    subplot.plot(df_result["lose"], label="lose")
    subplot.legend()

    if experiment is not None:
        experiment.log_figure(figure_name="win_vs_lose", figure=fig)

    fig = plt.figure(figsize=(20, 5))
    subplot = fig.add_subplot(111)
    subplot.plot(df_result["reward"], label="reward")
    subplot.legend()

    if experiment is not None:
        experiment.log_figure(figure_name="reward", figure=fig)

    fig = plt.figure(figsize=(20, 5))
    subplot = fig.add_subplot(111)
    subplot.plot(df_result["assets"], label="assets")
    subplot.legend()

    if experiment is not None:
        experiment.log_figure(figure_name="assets", figure=fig)
