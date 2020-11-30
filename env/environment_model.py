import numpy as np


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        """
            Constructor for the Environment Model of the Reinforcement learning framework

        :param n_states: Number of states in the Environment
        :param n_actions: Number of possible actions in the Environment
        :param seed: A seed to control the random number generator (optional)
        """

        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

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

    def draw(self, state, action):
        """
            Method to draw a next_state randomly based on probability of transitioning from state when action is chosen

        :param state: Index of current state
        :param action: Action to be taken
        :return: next_state, reward
        """

        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward

    def get_prob_rewards(self):
        """
            Method to get the probabilities and rewards for the env.
            raises NotImplementedError() if the method is not implemented by the super class.

        :return: probabilities, rewards as numpy arrays
        """

        raise NotImplementedError()