import numpy as np


class Testbed(object):
    def __init__(self, agents, steps=1000):
        self._agents = agents
        self._steps = steps
        self._action_history = np.zeros((len(self._agents), steps))
        self._reward_history = np.zeros((len(self._agents), steps))

    def run(self, env):
        for a in range(len(self._agents)):
            env.restart()
            self._agents[a].reset()
            for t in range(self._steps):
                action = self._agents[a].get_action()
                reward = env.reward(action)
                self._agents[a].update(action, reward)
                self._action_history[a, t] = action
                self._reward_history[a, t] = reward
