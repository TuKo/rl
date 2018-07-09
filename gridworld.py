import numpy as np


class GridWorld(object):
    def __init__(self, shape, gamma):
        self._grid = np.zeros(shape)
        self._gamma = gamma

        self._terminal = [0, np.prod(shape) - 1]
        self._cols = shape[1]
        self._rows = shape[0]

    def get_terminal_states(self):
        return self._terminal

    def reward(self, state, action):
        return -1

    def get_rewards(self):
        return [-1]

    def get_actions(self, state):
        return [-self._cols, -1, +1, +self._cols]

    def get_random_actions(self, states):
        possible_actions = np.array([-self._cols, -1, +1, +self._cols])
        actions = np.random.randint(4, size=len(states))
        return possible_actions[actions]

    def get_states(self):
        return np.arange(0, self._rows*self._cols)

    def get_probability(self, s, r, s_orig, action):
        states = self.get_states()
        if r != -1.0 or s not in states or s_orig not in states:
            return 0.0

        if s==s_orig:
            if  (s % self._cols == self._cols-1 and action == +1) or \
                (s % self._cols == 0 and action == -1) or \
                (s < self._cols and action == -self._rows) or \
                (s >= self._cols*(self._rows-1) and action == +self._rows):
                return 1.0
            else:
                return 0.0

        if action==(s-s_orig):
            if  (s < 0) or \
                (s >= self._cols * self._rows) or \
                (s % self._cols == 0 and action == +1) or \
                (s % self._cols == self._cols-1 and action == -1):
                return 0.0
            else:
                return 1.0

        return 0.0