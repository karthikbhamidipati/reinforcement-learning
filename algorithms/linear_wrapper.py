import numpy as np


class LinearWrapper:
    """
        Wrapper for env to perform Linear Value function approximation
    """

    def __init__(self, env):
        """
            Constructor for LinearWrapper

        :param env: Reinforcement learning environment
        """

        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        """
            Method for encoding a state into a feature matrix.
            Algorithm:
                1. Initialize features of size (num_actions, num_actions * n_states)
                2. For each action, calculate the flat index of the features for the state and action
                3. Mark the features matrix with the action index and above calculated index as 1.0

        :param s: State for which encoding should be performed
        :return: features of the encoded state
        """

        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        """
            Method to decode the theta and extract the policy and value

        :param theta: weight
        :return: policy and value decoded
        """

        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        """
            Method to reset the environment to the starting state encoded into feature matrix

        :return: starting state encoded into a feature matrix
        """

        return self.encode_state(self.env.reset())

    def step(self, action):
        """
            Method to call the step function of the environment and encode the next state

        :param action: action to be taken from the current state
        :return: encoded state, reward, done
        """

        state, reward, done = self.env.step(action)
        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        """
            Method to render the environment

        :param policy: Policy for the environment
        :param value: Value for the environment
        :return: None
        """

        self.env.render(policy, value)
