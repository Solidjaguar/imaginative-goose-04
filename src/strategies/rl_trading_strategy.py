import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

logger = logging.getLogger(__name__)

class TradingEnvironment(Env):
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.reward_range = (-np.inf, np.inf)
        self.action_space = Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(data.columns),), dtype=np.float32)
        
        self.current_step = 0
        self.total_steps = len(data)
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares = 0
        self.current_price = 0

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        return self._next_observation()

    def step(self, action):
        self._take_action(action)
        
        self.current_step += 1
        done = self.current_step >= self.total_steps
        
        obs = self._next_observation()
        reward = self._calculate_reward(action)
        info = {}
        
        return obs, reward, done, info

    def _next_observation(self):
        return self.data.iloc[self.current_step].values

    def _take_action(self, action):
        self.current_price = self.data.iloc[self.current_step]['Close']
        
        if action == 1:  # Buy
            shares_to_buy = self.balance // self.current_price
            self.shares += shares_to_buy
            self.balance -= shares_to_buy * self.current_price
        elif action == 2:  # Sell
            self.balance += self.shares * self.current_price
            self.shares = 0

    def _calculate_reward(self, action):
        if self.current_step > 0:
            price_change = self.current_price - self.data.iloc[self.current_step - 1]['Close']
            if action == 1:  # Buy
                return price_change if price_change > 0 else -abs(price_change) * 2
            elif action == 2:  # Sell
                return -price_change if price_change < 0 else -abs(price_change) * 2
            else:  # Hold
                return 0
        return 0

class RLTradingStrategy:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load_model(model_path)

    def train(self, data, total_timesteps=10000):
        env = DummyVecEnv([lambda: TradingEnvironment(data)])
        self.model = PPO("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
        logger.info(f"RL model trained for {total_timesteps} timesteps")

    def generate_signal(self, observation):
        if self.model is None:
            logger.error("Model not trained or loaded. Please train or load a model first.")
            return 0  # Hold
        
        action, _ = self.model.predict(observation)
        return action

    def save_model(self, filepath):
        if self.model is not None:
            self.model.save(filepath)
            logger.info(f"RL model saved to {filepath}")
        else:
            logger.error("No model to save. Please train a model first.")

    def load_model(self, filepath):
        self.model = PPO.load(filepath)
        logger.info(f"RL model loaded from {filepath}")

# You can add more RL-related functions here as needed