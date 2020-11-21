from rl.algorithms.modelbased.tabular_algorithms import policy_iteration, value_iteration
from rl.algorithms.modelfree.linear_wrapper import LinearWrapper
from rl.algorithms.modelfree.non_tabular_algorithms import linear_q_learning, linear_sarsa
from rl.algorithms.modelfree.tabular_algorithms import sarsa, q_learning
from rl.environment.frozenlake.frozenlake_environment import FrozenLake
from rl.environment.gridworld.gridworld_environment import GridWorld


def test_algorithms(env, gamma, theta, max_iterations):
    """
        TODO Clean this up

    :param env:
    :param gamma:
    :param theta:
    :param max_iterations:
    :return:
    """

    def print_values(policy, value, string):
        print(string)
        env.render(policy, value)
        print()
        print()

    wrapper = LinearWrapper(env)

    print_values(*value_iteration(env, gamma, theta, max_iterations), 'Value Iteration')
    print_values(*policy_iteration(env, gamma, theta, max_iterations), 'Policy Iteration')
    print_values(*sarsa(env, 1000, 1, gamma, 1), 'Sarsa')
    print_values(*q_learning(env, 1000, 1, gamma, 1), 'Q Learning')
    print_values(*wrapper.decode_policy(linear_sarsa(wrapper, 1000, 1, gamma, 1)), 'Linear Sarsa')
    print_values(*wrapper.decode_policy(linear_q_learning(wrapper, 1000, 1, gamma, 1)), 'Linear Q Learning')


def test_gridworld():
    grid = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '£'],
            ['#', '.', '.', '$']]

    test_algorithms(GridWorld(grid, 30), 0.9, 0, 10)


def test_frozenlake():
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    test_algorithms(FrozenLake(lake, 0.1, 30), 0.9, 0, 10)


test_frozenlake()
