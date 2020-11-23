import numpy as np


def argmax_random(actions):
    """
        Method to select an action that maximizes action values
        Algorithm:
            1. Get the maximum action value
            2. Get the actions that match the maxim value found in step 1
            3. Choose a random action from the array of actions found in step 2

    :param actions: action values generated by linear combination of features (action value pair of action and states) with weight/theta
    :return: an action after exploration
    """

    max_value = np.max(actions)
    max_indices = np.flatnonzero(max_value == actions)
    return np.random.choice(max_indices)


class EpsilonGreedySelection:
    """
        Class to select action based on epsilon greedy algorithm
    """

    def __init__(self, epsilon, random_state):
        """
            TODO ADD logic for seed

        :param random_state: random state for the game
        :param epsilon     : probability of choosing to explore
        """

        self.epsilon = epsilon
        self.random_state = random_state

    def select(self, action_values):
        """
            Method to select an action
            Algorithm:
                If the number generated by uniform distribution between 0 and 1 is less than epsilon,
                   then generate a random number between 0 and 3 for the four actions
                Else select an action that maximizes action values

        :param action_values: action values generated by linear combination of features (action value pair of action and states) with weight/theta
        :return: action selected by epsilon-greedy
        """

        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, len(action_values))
        else:
            return argmax_random(action_values)