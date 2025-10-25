import numpy as np
import gymnasium as gym


class FlattenObservation(gym.ObservationWrapper):
    """Flatten a (window, features) observation into 1D vector for agents that expect 1D observations.

    If the observation is already 1D it is returned unchanged.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        if hasattr(obs_space, 'shape') and len(obs_space.shape) > 1:
            new_shape = (int(np.prod(obs_space.shape)),)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=new_shape, dtype=np.float32)
        else:
            self.observation_space = obs_space

    def observation(self, obs):
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim > 1:
            return arr.ravel()
        return arr


class RewardNormalizer(gym.RewardWrapper):
    """A simple online reward normalizer using running mean/std.

    This helps stabilize RL training by keeping reward scale near 0/1.
    Not state-of-the-art but easy to integrate. You can persist the
    running stats and reuse them later.
    """
    def __init__(self, env, gamma: float = 0.99, eps: float = 1e-8):
        super().__init__(env)
        self.gamma = gamma
        self.eps = eps
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 1e-4
        self.returns = 0.0

    def reward(self, reward):
        # update discounted return
        self.returns = self.returns * self.gamma + reward
        # Welford online update on returns
        self.count += 1
        delta = self.returns - self.running_mean
        self.running_mean += delta / self.count
        delta2 = self.returns - self.running_mean
        self.running_var += delta * delta2
        std = (self.running_var / self.count) ** 0.5
        return reward / (std + self.eps)

    def reset(self, **kwargs):
        self.returns = 0.0
        return super().reset(**kwargs)
