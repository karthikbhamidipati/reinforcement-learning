import numpy as np


def argmax_random(actions):
    """
        TODO Add Description

    :param actions:
    :return:
    """

    max_value = np.max(actions)
    max_indices = np.flatnonzero(max_value == actions)
    return np.random.choice(max_indices)


class EpsilonGreedySelection:
    """
        TODO Add class Description
    """

    def __init__(self, epsilon, random_state):
        """
            TODO Add Description
            TODO ADD logic for seed

        :param random_state:
        :param epsilon:
        """

        self.epsilon = epsilon
        self.random_state = random_state

    def select(self, action_values):
        """
            TODO Add Description

        :param action_values:
        :return:
        """

        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, len(action_values))
        else:
            return argmax_random(action_values)
