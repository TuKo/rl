import numpy as np


class Greedy(object):
    def __init__(self, epsilon=0.0, k=10, rounds=1, Qinit=None):
        self._eps = epsilon
        self._rand_state = np.random.RandomState()
        self._k = k
        if Qinit is None:
            self._Qinit = np.zeros((k,))
        else:
            self._Qinit = Qinit
        self._Q = np.copy(self._Qinit)
        self._count = np.zeros((k,))
        self._cum_count = 0
        self._rounds = rounds

    def update(self, action, reward):
        self._count[action] += 1
        self._Q[action] += (reward - self._Q[action]) / self._count[action]
        self._cum_count += 1

    def get_action(self):
        # initialization steps
        # if self._cum_count < self._rounds*self._k:
        #     action = self._cum_count % self._k
        # Exploration?
        if self._rand_state.random_sample() < self._eps:
            action = self._rand_state.randint(self._k)
        else:
            action = np.argmax(self._Q)
        return action

    def reset(self):
        self._Q = np.copy(self._Qinit)
        self._count = np.zeros((self._k,))
        self._cum_count = 0
