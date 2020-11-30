from rl.algorithms.modelbased.tabular_algorithms import policy_iteration, value_iteration
from rl.algorithms.modelfree.linear_wrapper import LinearWrapper
from rl.algorithms.modelfree.non_tabular_algorithms import linear_q_learning, linear_sarsa
from rl.algorithms.modelfree.tabular_algorithms import sarsa, q_learning
from rl.environment.frozenlake.frozenlake_environment import FrozenLake

def main_implementation():
    seed = 0

    # Small lake
    smalllake = [['&', '.', '.', '.'],
                 ['.', '#', '.', '#'],
                 ['.', '.', '.', '#'],
                 ['#', '.', '.', '$']]

    env = FrozenLake(smalllake, slip=0.1, max_steps=16, seed=seed)

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 10

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 1000
    eta = 0.5
    epsilon = 0.5

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

def biglake_implementation():
    seed = 0

    biglake = [['&', '.', '.', '.','.','.','.','.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '#', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '#', '#', '.', '.', '.', '#', '.'],
            ['.', '#', '.', '.', '#', '.', '#', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '$']]

    env = FrozenLake(biglake, slip=0.1, max_steps=16, seed=seed)

    print('\n# Big lake implementation\n')

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 14


    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')


    print('# Model-free algorithms')
    max_episodes = 700000
    eta = 0.90
    epsilon = 0.99

    print('')

    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

    print('## Q-learning')
    max_episodes = 400000
    eta = 0.88
    epsilon = 0.98

    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)

    print('')

main_implementation()
biglake_implementation()