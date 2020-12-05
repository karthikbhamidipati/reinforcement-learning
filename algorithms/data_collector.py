import numpy as np


class DataCollectorSingleton(object):
    """
        Singleton Helper Class for collecting policy and rewards and calculating the errors.
    """

    ''' Variable to store Singleton instance of the class DataCollectorSingleton '''
    _instance = None

    def __init__(self):
        """
            Overriding the __init__ method to avoid direct instantiation
            throws RuntimeError if an object is created directly
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
        """
            Method to read the optimal policy file and set the optimal params

        :param environment: environment name
        :param file_name: file containing the optimal policy and value
        :return: None
        """

        optimal_policy_value = np.ndarray.tolist(np.load(file_name, allow_pickle=True))

        self.reset()

        self.optimal_env_params['environment'] = environment
        self.optimal_env_params['value'] = optimal_policy_value['value']
        self.optimal_env_params['policy'] = optimal_policy_value['policy']

    def calculate_error(self, algorithm, policy, value):
        """
            Method to calculate the policy and value error

        :param algorithm: Algorithm to be used as a key
        :param policy: Current policy from the algorithm
        :param value: Current value from the algorithm
        :return: None
        """

        if self.optimal_env_params['environment'] is not None:
            self._calculate_policy_error(algorithm, policy)
            self._calculate_value_error(algorithm, value)
        else:
            print("environment not set")

    def get_errors(self):
        """
            Method to get the errors calculated

        :return: Policy and value error
        """

        return self.policy_error, self.value_error

    def _calculate_policy_error(self, algorithm, policy):
        """
            Method to calculate the error in the policy
            Compares with the optimal policy and returns the average number of wrong policies

        :param algorithm: Algorithm to be used as a key
        :param policy: Policy from the environment
        :return: None
        """

        optimal_policy = self.optimal_env_params['policy']
        error = np.mean(np.not_equal(policy, optimal_policy))
        key = (self.optimal_env_params['environment'], algorithm)

        if key in self.value_error:
            self.policy_error[key].append(error)
        else:
            self.policy_error[key] = [error]

    def _calculate_value_error(self, algorithm, value):
        """
            Method to calculate the error in the value
            Compares with the optimal value and returns the mean squared error

        :param algorithm: Algorithm to be used as a key
        :param value: Value from the environment
        :return: None
        """

        optimal_value = self.optimal_env_params['value']
        error = np.mean(np.power(value - optimal_value, 2))
        key = (self.optimal_env_params['environment'], algorithm)

        if key in self.value_error:
            self.value_error[key].append(error)
        else:
            self.value_error[key] = [error]

    def reset(self):
        """
            Method to reset the stored data in memory

        :return: None
        """

        self.optimal_env_params.clear()
        self.value_error.clear()
        self.policy_error.clear()
