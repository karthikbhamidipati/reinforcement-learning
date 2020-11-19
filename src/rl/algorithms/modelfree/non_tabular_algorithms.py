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

        s, features = env.reset()
        q = features.dot(theta)

        done = False
        e_selection = EpsilonGreedySelection(epsilon[i], random_state)

        while not done:
            a = e_selection.select(q)
            s_prime, r, done = env.step(a)
            delta = r - q[a]

            features_prime = env.encode_state(s_prime)
            q_prime = features_prime.dot(theta)

            delta += gamma * np.max(q_prime)
            theta += eta[i] * delta * features[a, :]
            s = s_prime

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

        # TODO:

    return theta
