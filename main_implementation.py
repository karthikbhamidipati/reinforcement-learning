from algorithms.linear_wrapper import LinearWrapper
from algorithms.model_based_tabular_algorithms import policy_iteration, value_iteration
from algorithms.model_free_non_tabular_algorithms import linear_sarsa, linear_q_learning
from algorithms.model_free_tabular_algorithms import q_learning, sarsa
from algorithms.data_collector import DataCollectorSingleton
from env.frozenlake_environment import FrozenLake
from plot_episodes import plot_errors


def small_lake_implementation():
    seed = 0

    # Small lake
    small_lake = [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]

    env = FrozenLake(small_lake, slip=0.1, max_steps=16, seed=seed)

    gamma = 0.9
    theta = 0.0001
    max_iterations = 10
    DataCollectorSingleton.instance().set_optimal_policy_value("small_lake", "data/small_frozenlake_optimal_policy_value.npy")

    print('')

    # print('## Policy iteration')
    # policy, value = policy_iteration(env, gamma, theta, max_iterations)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('## Value iteration')
    # policy, value = value_iteration(env, gamma, theta, max_iterations)
    # env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    eta = 0.5
    epsilon = 0.5
    max_episodes = 2000

    print('')

    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')
    parameters = linear_sarsa(linear_env, max_episodes, eta,
                              gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')
    parameters = linear_q_learning(linear_env, max_episodes, eta,
                                   gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    plot_errors(*DataCollectorSingleton.instance().get_errors())


def big_lake_implementation():
    seed = 0

    big_lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '#', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '#', '#', '.', '.', '.', '#', '.'],
                ['.', '#', '.', '.', '#', '.', '#', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '$']]

    env = FrozenLake(big_lake, slip=0.1, max_steps=16, seed=seed)

    print('\n# Big lake implementation\n')

    DataCollectorSingleton.instance().set_optimal_policy_value("big_lake",
                                                               "data/big_frozenlake_optimal_policy_value.npy")
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.0001
    max_iterations = 19

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    max_iterations = 19
    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    # print('# Model-free algorithms')
    # max_episodes = 1000000
    # eta = 0.80
    # epsilon = 0.99
    #
    # print('')
    #
    # print('## Sarsa')
    # policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('## Q-learning')
    # max_episodes = 800000
    # eta = 0.88
    # epsilon = 0.90
    #
    # policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    # env.render(policy, value)
    #
    # print('')
    plot_errors(*DataCollectorSingleton.instance().get_errors())


small_lake_implementation()
# big_lake_implementation()
