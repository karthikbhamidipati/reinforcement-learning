import numpy as np

from rl.algorithms.modelfree.epsilon_greedy import EpsilonGreedySelection


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
        TODO Write working code

    :param env:
    :param max_episodes:
    :param eta:
    :param gamma:
    :param epsilon:
    :param seed:
    :return:
    """

    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        e_selection = EpsilonGreedySelection(epsilon[i], random_state)
        q = np.dot(features, theta)
        a = e_selection.select(q)
        done = False

        while not done:
            features_prime, r, done = env.step(a)
            delta = r - q[a]
            q = np.dot(features_prime, theta)
            a_prime = e_selection.select(q)
            delta += gamma * q[a_prime]
            theta += eta[i] * delta * features[a]
            features = features_prime
            a = a_prime

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
        TODO Write working code

    :param env:
    :param max_episodes:
    :param eta:
    :param gamma:
    :param epsilon:
    :param seed:
    :return:
    """

    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        e_selection = EpsilonGreedySelection(epsilon[i], random_state)
        q = np.dot(features, theta)
        done = False

        while not done:
            a = e_selection.select(q)
            features_prime, r, done = env.step(a)
            delta = r - q[a]
            q = np.dot(features_prime, theta)
            delta += gamma * np.max(q)
            theta += eta[i] * delta * features[a]
            features = features_prime

    return theta
