import numpy as np

from rl.algorithms.modelfree.epsilon_greedy import EpsilonGreedySelection


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
        TODO Add Documentation

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

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        e_selection = EpsilonGreedySelection(epsilon[i], random_state)
        a = e_selection.select(q[s])
        done = False

        while not done:
            s_prime, r, done = env.step(a)
            a_prime = e_selection.select(q[s_prime])
            q[s, a] += eta[i] * (r + (gamma * q[s_prime, a_prime]) - q[s, a])
            s = s_prime
            a = a_prime

    policy = np.argmax(q, axis=1)
    value = np.max(q, axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
        TODO Add Documentation

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

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        e_selection = EpsilonGreedySelection(epsilon[i], random_state)
        a = e_selection.select(q[s])
        done = False

        while not done:
            s_prime, r, done = env.step(a)
            a_prime = e_selection.select(q[s_prime])
            q[s, a] += eta[i] * (r + (gamma * np.max(q[s_prime])) - q[s, a])
            s = s_prime
            a = a_prime

    policy = np.argmax(q, axis=1)
    value = np.max(q, axis=1)

    return policy, value