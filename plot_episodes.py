import matplotlib.pyplot as plt
import numpy as np


def plot_errors(policy_error, value_error):
    def _plot_data(param, value, error_type):
        episodes = np.arange(len(value)) + 1
        plt.plot(episodes, value)
        plt.xlabel("Episodes/Iterations")
        plt.ylabel(error_type)
        plt.title("{} vs Episodes/Iterations\nfor {} environment, {} algorithm ".format(error_type, *param))
        plt.show()

    for key in value_error.keys():
        value_data = value_error[key]
        policy_data = policy_error[key]

        _plot_data(key, value_data, "Value error (mse)")
        _plot_data(key, policy_data, "Policy error")
