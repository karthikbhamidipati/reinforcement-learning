import matplotlib.pyplot as plt
import numpy as np


class DataCollectorSingleton(object):
    """
        # TODO
        Singleton Helper Class for calculating policy & rewards only once per env.
    """

    ''' Variable to store Singleton instance of the class DataCollectorSingleton '''
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
            Static method to create(if doesn't exist) and return an instance of the class DataCollectorSingleton
        :return: instance of the class DataCollectorSingleton
        """

        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            cls.optimal_env_params = {}
            cls.value_error = {}
            cls.policy_error = {}

        return cls._instance

    def set_optimal_policy_value(self, environment, file_name):
        optimal_policy_value = np.ndarray.tolist(np.load(file_name, allow_pickle=True))

        self.reset()

        self.optimal_env_params['environment'] = environment
        self.optimal_env_params['value'] = optimal_policy_value['value']
        self.optimal_env_params['policy'] = optimal_policy_value['policy']

    def calculate_error(self, algorithm, policy, value):
        if self.optimal_env_params['environment'] is not None:
            self._calculate_policy_error(algorithm, policy)
            self._calculate_value_error(algorithm, value)
        else:
            print("environment not set")

    def get_errors(self):
        return self.policy_error, self.value_error

    def _calculate_policy_error(self, algorithm, policy):
        optimal_policy = self.optimal_env_params['policy']
        error = np.mean(np.equal(policy, optimal_policy))
        key = (self.optimal_env_params['environment'], algorithm)

        if key in self.value_error:
            self.policy_error[key].append(error)
        else:
            self.policy_error[key] = [error]

    def _calculate_value_error(self, algorithm, value):
        optimal_value = self.optimal_env_params['value']
        error = np.mean(np.power(value - optimal_value, 2))
        key = (self.optimal_env_params['environment'], algorithm)

        if key in self.value_error:
            self.value_error[key].append(error)
        else:
            self.value_error[key] = [error]

    def reset(self):
        self.optimal_env_params.clear()
        self.value_error.clear()
        self.policy_error.clear()
