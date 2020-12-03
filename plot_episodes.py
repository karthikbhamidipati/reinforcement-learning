import matplotlib.pyplot as plt
import numpy as np

def plot_graph(data_npy_file):
    # Loading data from .npy file
    data = np.load(data_npy_file, allow_pickle=True)
    data = np.ndarray.tolist(data)

    max_episodes  = data.get("max_episodes")

    avg_return = data.get("avg_return")
    #avg_theta    = data.get("avg_theta")

    episodes = np.arange(max_episodes)
    plt.plot(episodes, avg_return)
    #plt.plot(episodes, avg_theta)
    plt.xlabel("Episodes")
    plt.ylabel("Average Return")
    #plt.ylabel("Average Theta")
    plt.show()

#data_npy_file = 'data/small_frozenlake_sarsa.npy'
#data_npy_file = 'data/small_frozenlake_q_learning.npy'
#data_npy_file = 'data/small_frozenlake_linear_sarsa.npy'
#data_npy_file = 'data/small_frozenlake_linear_q_learning.npy'
data_npy_file = 'data/big_frozenlake_sarsa.npy'
#data_npy_file = 'data/big_frozenlake_q_learning.npy'
plot_graph(data_npy_file)