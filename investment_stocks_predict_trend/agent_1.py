import pandas as pd
import numpy as np
from comet_ml import Experiment
import chainer
import chainerrl
import matplotlib.pyplot as plt


def build_experiment(project_name="learn-stocks.2"):
    experiment = Experiment(api_key=COMET_ML_API_KEY, project_name=project_name)

    return experiment


def end_experiment(experiment):
    experiment.end()




def preprocessing():
    df_csv = pd.read_csv("local/nikkei_averages.csv")
    df_csv.info()
    print(df_csv.head())
    print(df_csv.tail())
    
    df = df_csv.copy()
    df = df[["date", "opening_price", "high_price", "low_price", "close_price"]]
    df = df.sort_values("date")
    df = df.drop_duplicates()
    df = df.assign(id=np.arange(len(df)))
    df = df.set_index("id")
    
    df = df.assign(rate_of_return=df["close_price"].pct_change())
    df.info()
    print(df.head())
    print(df.tail())

    return df




class LearnEnv():
  def __init__(self, df, start_id, end_id):
    self.DF = df.copy()
    self.START_ID = start_id
    self.END_ID = end_id

    self.reset()
    
    self.data_len = self.END_ID - self.START_ID
    self.action_size = 2 # 0...何もしない、1...購入or売却
    self.observation_size = len(self.observe())
    
  def reset(self):
    self.total_reward = 0.0
    self.funds = 0.0
    self.current_id = self.START_ID
    self.buy_price = 0.0
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
      self.buy_price = self.df_action.at[self.current_id, "opening_price"]
      self.funds -= self.buy_price
      reward = 0.0
      
      self.df_action.at[self.current_id, "buy"] = 1
    elif self.buy_price != 0.0:
      # sell
      sell_price = self.df_action.at[self.current_id, "close_price"]
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
        [self.df_action.at[self.current_id - i, "rate_of_return"] for i in range(1, 6)],
        dtype=np.float32
    )
    
    return obs
  
  def random_action(self):
    return np.random.randint(0, 2)



def build_env():
    env = LearnEnv(df, 19090-250, 19090)

    return env



def build_agent(env, experiment):
    hyper_params = {
        "n_hidden_layers": 3,
        "obs_size": env.observation_size,
        "n_actions": env.action_size,
        "n_hidden_channels": env.observation_size * env.action_size,
        "adam_eps": 1e-2,
        "gamma": 0.95,
        "start_epsilon": 1.0,
        "end_epsilon": 0.3,
        "decay_steps": 200 * 200,
        "replay_buffer_capacity": 10 ** 6,
        "ddqn_replay_start_size": 500,
        "ddqn_update_interval": 1,
        "ddqn_target_update_interval": 100
    }

    q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        hyper_params["obs_size"],
        hyper_params["n_actions"],
        n_hidden_layers=hyper_params["n_hidden_layers"],
        n_hidden_channels=hyper_params["n_hidden_channels"])
    q_func.to_gpu(0)

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
        target_update_interval=hyper_params["ddqn_target_update_interval"])

    return agent



def learn_agent(env, agent, experiment):
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
      experiment.log_metrics(metrics, step=i)
      
      if i % 10 == 0:
        print("episode:", i, ", R:", R, ", statistics:", agent.get_statistics(), ", epsilon:", agent.explorer.epsilon)
        env.render()




def simulate_agent(env, agent, experiment):
    obs = env.reset()
    R = 0
    done = False

    while not done:
      action = agent.act(obs)
      obs, reward, done, _ = env.step(action)
      
      env.render()

    agent.stop_episode()

    df_result = env.df_action.query("18840 <= id <= 19090").copy()

    experiment.log_asset_data(df_result.to_csv(), file_name="result.csv")

    df_result




def build_figure_win_vs_lose(df_result, experiment):
    fig = plt.figure(figsize=(20, 5))
    subplot = fig.add_subplot(111)
    subplot.plot(df_result["win"], label="win")
    subplot.plot(df_result["lose"], label="lose")
    subplot.legend()

    plt.show()

    experiment.log_figure(figure_name="win_vs_lose", figure=fig)



def build_figure_reward(df_result, experiment):
    fig = plt.figure(figsize=(20, 5))
    subplot = fig.add_subplot(222)
    subplot.plot(df_result["reward"], label="reward")
    subplot.legend()

    plt.show()

    experiment.log_figure(figure_name="reward", figure=fig)




