import numpy as np


class PolicyRewardSingleton(object):
    """
        Singleton Helper Class for calculating policy & rewards only once per env.
    """

    ''' Variable to store Singleton instance of the class PolicyRewardSingleton '''
    _instance = None

    def __init__(self):
        """
            Overriding the __init__ method to avoid direct instantiation
            throws RuntimeError if an object is created directly.
        """

        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        """
            Static method to create(if doesn't exist) and return an instance of the class PolicyRewardSingleton

        :return: instance of the class PolicyRewardSingleton
        """

        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            cls.env_dict = {}
        return cls._instance

    def get_prob_rewards(self, env):
        """
            Method to get the probability & rewards for env
            Calculates the probability & rewards for the env if they're not already calculated

        :param env: Environment for which the probability & rewards need to be returned
        :return: probability & rewards for env as a tuple
        """

        if env not in self.env_dict:
            self._calc_prob_rewards(env)

        return self.env_dict[env]

    def _calc_prob_rewards(self, env):
        """
            Private method for updating the env_dict with probability & rewards for the env
            TODO

        :param env: Environment for which the probability & rewards need to be calculated
        :return: None
        """

        p = np.empty([env.n_states, env.n_states, env.n_actions])
        r = np.empty_like(p)

        for s in range(env.n_states):
            for s_prime in range(env.n_states):
                for action in range(env.n_actions):
                    p[s, s_prime, action] = env.p(s_prime, s, action)
                    r[s, s_prime, action] = env.r(s_prime, s, action)

        self.env_dict[env] = (p, r)
