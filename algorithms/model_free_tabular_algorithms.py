import numpy as np

from algorithms.data_collector import DataCollectorSingleton
from algorithms.epsilon_greedy import EpsilonGreedySelection


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
            Method to implement SARSA control
            Algorithm:
                1. initialization:
                           - generate a random state for the game
                           - create evenly spaced learning rates over maximum episodes
                           - create evenly spaced exploration factor over maximum episodes
                           - action value function, initialized to zeroes of 2D array size - n_states * n_actions
                           [[0 0 0 0]   --- state 1
                            .  . . .
                            .
                            .
                            [0 0 0 0]]  --- state 17 (one extra state including the absorbing state)
                2. for each episode:
                           - initialise the state for the episode i
                           - select an action from the random state with the epsilon-greedy policy
                           - execute step 3 till the end of game
                3. while the terminal state is not reached:
                           - get the reward(r), game state(done) and next state (s')  for the selected action
                             in the current state
                           - Select action a′ for state s′ according to an ε-greedy policy based on Q
                           - calculate the action value for the current state and action:
                             Q(s, a) ← Q(s, a) + α[r + γQ(s′, a′) − Q(s, a)]
                           - re-assign the next state and action to the current state and action for the next iteration
                           s ← s′
                           a ← a'
                4. Get the policy and value that maximizes the action value computed in step 3

    :param env:          Environment of the game
    :param max_episodes: Maximum number of episodes
    :param eta:          Learning rate (α)
    :param gamma:        Discount factor (γ)
    :param epsilon:      Exploration factor
    :param seed:         Pseudorandom number generator
    :return:             policy and value
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

        DataCollectorSingleton.instance().calculate_error("Sarsa", np.argmax(q, axis=1), np.max(q, axis=1))

    policy = np.argmax(q, axis=1)
    value = np.max(q, axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
            Method to implement Q-learning control
            Algorithm:
                1. initialization:
                           - generate a random state for the game
                           - create evenly spaced learning rates over maximum episodes
                           - create evenly spaced exploration factor over maximum episodes
                           - action value function, initialized to zeroes of 2D array size - n_states * n_actions
                           [[0 0 0 0]   --- state 1
                            .  . . .
                            .
                            .
                            [0 0 0 0]]  --- state 17 (one extra state including the absorbing state)
                2. for each episode:
                           - initialise the state for the episode i
                           - select an action from the random state with the epsilon-greedy policy
                           - execute step 3 till the end of game
                3. while the terminal state is not reached:
                           - get the reward(r), game state(done) and next state (s')  for the selected action
                             in the current state
                           - Select action a' for state s' according to an ε-greedy policy based on Q
                           - calculate the action value for the current state and action:
                             Q(s,a)←Q(s,a)+α[r+γ maxa′Q(s′,a′)−Q(s,a)]
                           - re-assign the next state and action to the current state and action for the next iteration
                           s ← s′
                           a ← a'
                4. Get the policy and value that maximizes the action value computed in step 3

    :param env:          Environment of the game
    :param max_episodes: Maximum number of episodes
    :param eta:          Learning rate (α)
    :param gamma:        Discount factor (γ)
    :param epsilon:      Exploration factor
    :param seed:         Pseudorandom number generator
    :return:             policy and value
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

        DataCollectorSingleton.instance().calculate_error("Q-learning", np.argmax(q, axis=1), np.max(q, axis=1))

    policy = np.argmax(q, axis=1)
    value = np.max(q, axis=1)

    return policy, value
