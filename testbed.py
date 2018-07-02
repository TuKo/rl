import numpy as np


class Testbed(object):
    def __init__(self, agents, steps=1000, runs=2000):
        self._agents = agents
        self._steps = steps
        # for i in range(self._total_runs):
        # self._total_runs = runs
        # self._runs = 0
        self._action_history =
        self._reward_history = np.zeros((len(self._agents), steps))

    def run(self, env):
        for a in range(len(self._agents)):
            env.reset()
            for t in range(self._steps):
                action = self._agents[a].get_action()
                reward = env.reward(action)
                self._agents[a].update(action, reward)
                self._action_history[a, t] = action
                self._reward_history[a, t] = reward
