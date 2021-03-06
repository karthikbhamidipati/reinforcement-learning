import numpy as np


class EpsilonGreedySelection:
    """
        Class to select action based on epsilon greedy algorithm
    """

    def __init__(self, epsilon, random_state=None):
        """
            Constructor for EpsilonGreedySelection

        :param random_state: random state for the game
        :param epsilon     : probability of choosing to explore
        """

        self.epsilon = epsilon
        self.random_state = np.random.RandomState() if random_state is None else random_state

    def select(self, action_values):
        """
            Method to select an action
            Algorithm:
                1. Generate the random number using a uniform distribution between 0 and 1
                2. If the random number is less than epsilon, pick a random action
                3. Else, pick the greedy action that maximizes the value

        :param action_values: Action values represented in an vector
        :return: action selected by epsilon-greedy
        """

        if self.random_state.uniform(0, 1) < self.epsilon:
            return self.random_state.randint(0, len(action_values))
        else:
            return self.argmax_random(action_values)

    def argmax_random(self, actions):
        """
            Method to select an action that maximizes action values
            Algorithm:
                1. Get the maximum action value
                2. Get the actions that match the maximum value found in step 1
                4. Break ties at random and choose an action for the actions with maximum value in step 2

        :param actions: action values generated by linear combination of features (action value pair of action and states) with weight/theta
        :return: an action which maximizes the return
        """

        max_value = np.max(actions)
        max_indices = np.flatnonzero(max_value == actions)
        return self.random_state.choice(max_indices)
