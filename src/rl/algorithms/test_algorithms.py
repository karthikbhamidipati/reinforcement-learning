from rl.algorithms.tabular.model_based_algorithms import policy_iteration, value_iteration
from rl.environment.frozenlake.frozenlake_environment import FrozenLake
from rl.environment.gridworld.gridworld_environment import GridWorld


def test_algorithms(env, gamma, theta, max_iterations):
    def print_values(policy, value, string):
        print(string)
        env.render(policy, value)
        print()
        print()

    print_values(*value_iteration(env, gamma, theta, max_iterations), 'Value Iteration')
    print_values(*policy_iteration(env, gamma, theta, max_iterations), 'Policy Iteration')


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
