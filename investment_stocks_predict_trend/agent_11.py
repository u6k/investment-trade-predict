import pandas as pd
import numpy as np
import chainer
import chainerrl
import matplotlib.pyplot as plt


def execute(experiment=None, max_episode=500):
    df = preprocessing()
    print(df)

    train_env = LearnEnv(df, 18000, 18750)
    test_env = LearnEnv(df, 18750, 19000)

    agent = build_agent(train_env, experiment)

    for i in range(1, max_episode+1):
        print("*** episode: "+str(i)+" ***")
        df_result, metrics = train_agent(train_env, agent)
        if experiment is not None:
            experiment.log_asset_data(df_result.to_csv(), file_name="train_result."+str(i)+".csv")
            experiment.log_metrics(metrics, step=i)

        if i % 10 == 0:
            print("episode: "+str(i))
            print(metrics)

            df_result, metrics = simulate_agent(test_env, agent)
            if experiment is not None:
                experiment.log_asset_data(df_result.to_csv(), file_name="test_result."+str(i)+".csv")

            build_figure_result(df_result, experiment)


def preprocessing():
    df_csv = pd.read_csv("local/nikkei_averages.csv", index_col=0)

    df = df_csv.copy()

    df = df[["date", "opening_price", "high_price", "low_price", "close_price"]]
    df = df.sort_values("date")
    df = df.drop_duplicates()
    df = df.assign(id=np.arange(len(df)))
    df = df.set_index("id")

    df = df.assign(diff=(df["close_price"] - df["opening_price"]))

    return df


class LearnEnv():
    def __init__(self, df, start_id, end_id):
        self.DF = df.copy()
        self.START_ID = start_id
        self.END_ID = end_id

        self.reset()

        self.data_len = self.END_ID - self.START_ID
        self.action_size = 2  # 0...何もしない、1...購入and売却
        self.observation_size = len(self.observe())

    def reset(self):
        self.total_reward = 0.0
        self.funds = 0.0
        self.assets = 0.0
        self.current_id = self.START_ID
        self.done = False
        self.win = 0
        self.lose = 0

        self.df_action = self.DF.copy()
        self.df_action = self.df_action.assign(reward=0.)
        self.df_action = self.df_action.assign(funds=0.)
        self.df_action = self.df_action.assign(assets=0.)
        self.df_action = self.df_action.assign(action=0)
        self.df_action = self.df_action.assign(win=0)
        self.df_action = self.df_action.assign(lose=0)

        return self.observe()

    def step(self, action):
        if action == 0:
            reward = 0.0
        else:
            reward = self.df_action.at[self.current_id, "diff"]
            self.funds += reward
            self.assets = self.funds
            self.total_reward += reward

            if reward > 0:
                self.win += 1
            else:
                self.lose += 1

            self.df_action.at[self.current_id, "action"] = 1

        self.df_action.at[self.current_id, "reward"] = self.total_reward
        self.df_action.at[self.current_id, "funds"] = self.funds
        self.df_action.at[self.current_id, "assets"] = self.assets
        self.df_action.at[self.current_id, "win"] = self.win
        self.df_action.at[self.current_id, "lose"] = self.lose

        self.current_id += 1
        if self.current_id >= self.END_ID:
            self.done = True

        return self.observe(), reward, self.done, {}

    def render(self):
        print("id: "+str(self.current_id))
        print("total_reward: "+str(self.total_reward) +
              ", funds: "+str(self.funds) +
              ", assets: "+str(self.assets) +
              ", win: "+str(self.win) +
              ", lose: "+str(self.lose))
        print("observe:")
        print(self.observe())

    def observe(self):
        obs = np.array(
            [self.df_action.at[self.current_id-i, "diff"] for i in range(1, 21)],
            dtype=np.float32
        )

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

    env.render()

    while not done:
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)

        env.render()

    agent.stop_episode_and_train(obs, reward, done)

    metrics = {
        "reward": env.total_reward,
        "epsilon": agent.explorer.epsilon,
        "win": env.win,
        "lose": env.lose,
        "funds": env.funds,
        "assets": env.assets
    }

    df_result = env.df_action.query(str(env.START_ID)+" <= id <= "+str(env.END_ID)).copy()

    return df_result, metrics


def simulate_agent(env, agent):
    obs = env.reset()
    done = False

    env.render()

    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)

        env.render()

    agent.stop_episode()

    metrics = {
        "reward": env.total_reward,
        "epsilon": agent.explorer.epsilon,
        "win": env.win,
        "lose": env.lose,
        "funds": env.funds,
        "assets": env.assets
    }

    df_result = env.df_action.query(str(env.START_ID) + " <= id <= " + str(env.END_ID)).copy()

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
