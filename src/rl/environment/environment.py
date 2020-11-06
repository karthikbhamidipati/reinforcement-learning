import numpy as np

from rl.environment.environment_model import EnvironmentModel


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, dist, seed=None):
        super(Environment, self).__init__(n_states, n_actions, seed)

        self.max_steps = max_steps

        self.dist = dist
        if self.dist is None:
            self.dist = np.full(n_states, 1. / n_states)

        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.dist)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.dist)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid Action!!!')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)
        return self.state, reward, done

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()
