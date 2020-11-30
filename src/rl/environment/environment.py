import contextlib

import numpy as np

from rl.environment.environment_model import EnvironmentModel


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        """
            Constructor of the abstract base class for creating environments

        :param n_states: Number of states in the Environment
        :param n_actions: Number of possible actions in the Environment
        :param max_steps: The maximum number of time steps in an episode
        :param pi: Probability distribution for choosing the starting state
        :param seed: A seed to control the random number generator (optional)
        """

        super(Environment, self).__init__(n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

    def reset(self):
        """
            Method to reset the environment to a starting state
            Draws a random state using the probability distribution(pi) of starting states

        :return: Starting state of the environment after reset
        """

        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        return self.state

    def step(self, action):
        """
            Method to perform action on the current state
            Throws an exception if the action is not valid
            Uses draw() from EnvironmentModel class to identify the next state and reward

        :param action: Action to be taken from current state
        :return: the current state, reward & boolean value to indicate if the game is over
        """

        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid Action!!!')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        """
            Method to visualize the GridWorld
            To be implemented by subclasses
            Raises NotImplementedError() if render() is not implemented

        :param policy: policy to be rendered
        :param value: value to be rendered
        :return: None
        """

        raise NotImplementedError()

    def p(self, next_state, state, action):
        """
            Method to calculate probability of transitioning between state and next_state with action
            To be implemented by subclasses
            Raises NotImplementedError() if p() is not implemented

        :param next_state: Index of next state
        :param state: Index of current state
        :param action: Action to be taken
        :return: Probability of transitioning between state and next_state with action
        """

        raise NotImplementedError()

    def r(self, next_state, state, action):
        """
            Method to calculate reward of transitioning between state and next_state with action
            To be implemented by subclasses
            Raises NotImplementedError() if r() is not implemented

        :param next_state: Index of next state
        :param state: Index of current state
        :param action: Action to be taken
        :return: Reward for transitioning between state and next_state with action
        """

        raise NotImplementedError()

    def get_prob_rewards(self):
        """
            Method to get the probabilities and rewards for the env.
            raises NotImplementedError() if the method is not implemented by the super class.

        :return: probabilities, rewards as numpy arrays
        """

        raise NotImplementedError()

    @contextlib.contextmanager
    def _printoptions(self, *args, **kwargs):
        """
            Method to set the print options for numpy arrays based on the arguments passed

        :param args: Non Keyword Arguments
        :param kwargs: Keyword Arguments
        :return: None
        """

        original = np.get_printoptions()
        np.set_printoptions(*args, **kwargs)
        try:
            yield
        finally:
            np.set_printoptions(**original)
