import numpy as np


class Bandit(object):
    def __init__(self, k=10, mean=0.0, std=1.0):
        self._arms = k
        self._mean = mean
        self._std = std

        self._q_star = np.random.normal(loc=mean, scale=std, size=k)

    def reward(self, action):
        return np.random.normal(self._q_star[action], scale=self._std, size=1)

    def reset(self):
        pass
