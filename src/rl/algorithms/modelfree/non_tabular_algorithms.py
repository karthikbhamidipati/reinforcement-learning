import numpy as np

from rl.algorithms.modelfree.epsilon_greedy import EpsilonGreedySelection


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
            Method to implement linear approximation with SARSA control
            Algorithm:
                1. initialization:
                           - generate a random state for the game
                           - create evenly spaced learning rates over maximum episodes
                           - create evenly spaced exploration factor over maximum episodes
                           - weight/theta initialized to zeroes of size of feature vector
                2. for each episode:
                           - initialise the state for the episode i
                           - linearly combine the features (action value pair of action and states) with weight/theta
                             Q(a) ← 􏰀􏰀􏰀Σi θi φ(s, a)i
                           - select an action from the random state with the epsilon-greedy policy
                           - execute step 3 till the end of game
                3. while the terminal state is not reached:
                           - get the reward(r), game state(done) and features of the next state (φ(s', a')  for the selected action
                             in the current state
                           - calculate a part of the temporal difference, δ ← r − Q(a)
                           - linearly combine the next features (action value of next pair of action and states) with weight/theta
                             Q(a′) ← 􏰀Σi θi φ(s′, a′)i
                           - select next action from the current state with the epsilon-greedy policy
                           - calculate the temporal difference:
                             δ ← δ + γ * Q(a′) - Q(a)
                           - recalculate the weights/theta for all the features for the current state and action with the below equation:
                             θ ← θ + αδφ(s, a)
                           - re-assign the next set of features and action to the current state and features for the next iteration
                           s ← s′
                           φ(s, a) ← φ(s', a')

    :param env:          Environment of the game
    :param max_episodes: Maximum number of episodes
    :param eta:          Learning rate (α)
    :param gamma:        Discount factor (γ)
    :param epsilon:      Exploration factor
    :param seed:         Pseudorandom number generator
    :return:             Weights - the learnable parameter
    """

    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = np.dot(features, theta)
        e_selection = EpsilonGreedySelection(epsilon[i], random_state)
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
            Method to implement linear approximation with Q-learning control
            Algorithm:
                1. initialization:
                           - generate a random state for the game
                           - create evenly spaced learning rates over maximum episodes
                           - create evenly spaced exploration factor over maximum episodes
                           - weight/theta initialized to zeroes of size of feature vector
                2. for each episode:
                           - initialise the state for the episode i
                           - linearly combine the features (action value pair of action and states) with weight/theta
                             Q(a) ← 􏰀􏰀􏰀Σi θi φ(s, a)i
                           - execute step 3 till the end of game
                3. while the terminal state is not reached:
                           - select an action from the random state with the epsilon-greedy policy
                           - get the reward(r), game state(done) and features of the next state (φ(s', a')  for the selected action
                             in the current state
                           - calculate a part of the temporal difference, δ ← r − Q(a)
                           - linearly combine the next features (action value of next pair of action and states) with weight/theta
                             Q(a′) ← 􏰀Σi θi φ(s′, a′)i
                           - calculate the temporal difference:
                             δ ← δ + γ maxa′ Q(a′)
                           - recalculate the weights/theta for all the features for the current state and action with the below equation:
                             θ ← θ + αδφ(s, a)
                           - re-assign the next set of features and action to the current state and features for the next iteration
                           φ(s, a) ← φ(s', a')

    :param env:          Environment of the game
    :param max_episodes: Maximum number of episodes
    :param eta:          Learning rate (α)
    :param gamma:        Discount factor (γ)
    :param epsilon:      Exploration factor
    :param seed:         Pseudorandom number generator
    :return:             Weights - the learnable parameter
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
