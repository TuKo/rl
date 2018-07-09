import numpy as np


def compute_Bellman(env, v, s_orig, a):
    # we are in state s and are considering the action a
    states = env.get_states()
    rewards = env.get_rewards()
    table = np.zeros((len(states), len(rewards)))

    for s in states:
        for r_id, r in enumerate(rewards):
            table[s, r_id] = env.get_probability(s, r, s_orig, a) * (r + env._gamma * v[s])
    return np.sum(table)


class IterativePolicyEvaluation(object):
    # Note: this is actually not Iterative Policy Evaluation because it assumes a deterministic policy.
    # The policy should be stochastic and should be used to compute v[s] (see Eq. (4.5) in the book).
    def __init__(self, environment, epsilon=1e-4):
        self._env = environment
        self._eps = epsilon

    def init(self):
        states = self._env.get_states()
        v = np.zeros((len(states),))
        v[self._env.get_terminal_states()] = 0
        self.v = v
        return v

    def run(self, policy):
        if not hasattr(self, 'v'):
            self.init()
        v = self.v
        states = self._env.get_states()
        delta = 2*self._eps
        while delta > self._eps:
            delta = 0.0
            v_old = v.copy()
            for s in states:
                if s in self._env.get_terminal_states():
                    v[s] = 0.0
                    continue
                vs = v_old[s]
                v[s] = compute_Bellman(self._env, v, s, policy[s])
                delta = max(delta, abs(vs - v[s]))
            print('D ', delta)

        self.v = v
        return v


class PolicyImprovement(object):
    def __init__(self, environment):
        self._env = environment

    def init(self):
        states = self._env.get_states()
        self.policy = self._env.get_random_actions(states)
        return self.policy

    def run(self, v):
        if not hasattr(self, 'policy'):
            self.init()
        policy = self.policy
        policy_stable = True
        states = self._env.get_states()
        for s in states:
            old_action = policy[s]
            # output for argmax is vector of length |Actions|
            actions = self._env.get_actions(s)
            action_values = np.zeros((len(actions),))
            for a_id, a in enumerate(actions):
                action_values[a_id] = compute_Bellman(self._env, v, s, a)
            policy[s] = actions[np.argmax(action_values)]
            # TODO: Bug is here. May change policy but have the same state-value func.
            # Need to check the function that it is improved, following the policy improvement theorem.
            if old_action != policy[s]:
                policy_stable = False
        return policy, policy_stable


class PolicyIteration(object):
    def __init__(self, environment):
        self._eval = IterativePolicyEvaluation(environment)
        self._impr = PolicyImprovement(environment)

    def init(self):
        v = self._eval.init()
        policy = self._impr.init()
        return v, policy

    def run(self):
        v, policy = self.init()

        stable = False
        while not stable:
            v = self._eval.run(policy)
            print(v)
            policy, stable = self._impr.run(v)
            print(policy)

        self.policy = policy
        self.v = v
        return policy, v


class ValueIteration(object):
    def __init__(self, environment, eps=1e-4):
        self._env = environment
        self._eps = eps

    def init(self):
        states = self._env.get_states()
        v = np.zeros((len(states),))
        v[self._env.get_terminal_states()] = 0
        self.v = v
        return v

    def run(self):
        v = self.init()
        states = self._env.get_states()
        delta = 2*self._eps
        while delta > self._eps:
            delta = 0.0
            for s in states:
                if s in self._env.get_terminal_states():
                    v[s] = 0.0
                    continue
                vs = v[s].copy()
                actions = self._env.get_actions(s)
                action_values = np.zeros((len(actions),))
                for a_id, a in enumerate(actions):
                    action_values[a_id] = compute_Bellman(self._env, v, s, a)
                v[s] = np.max(action_values)
                delta = max(delta, abs(vs - v[s]))
            print('D ', delta)

        self.v = v
        print(v)

        # Find the best policy
        for s in states:
            actions = self._env.get_actions(s)
            action_values = np.zeros((len(actions),))
            for a_id, a in enumerate(actions):
                action_values[a_id] = compute_Bellman(self._env, v, s, a)
            policy[s] = actions[np.argmax(action_values)]
        self.policy = policy
        print(policy)


if __name__ == "__main__":
    from gridworld import *
    env = GridWorld((4, 4), 0.9)
    print(env.get_actions(1))
    print(env.get_terminal_states())
    print(env.get_random_actions([0, 1, 2, 3]))
    print(env.get_states())
    print(env.reward(0, -1))
    print(env.get_probability(7, -1, 3, 4))
    print('Policy Eval')
    states = env.get_states()
    policy = [0] * len(states)
    for s in states:
        actions = env.get_actions(s)
        action = actions[0]
        policy[s] = action
    print(policy)
    ipe = IterativePolicyEvaluation(env)
    print(ipe.run(policy))
    print(ipe.v)
    print('Policy Iteration')
    t = PolicyIteration(env)
    t.run()
    print('Value Iteration')
    v = ValueIteration(env)
    v.run()

