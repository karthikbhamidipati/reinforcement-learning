from rl.algorithms.tabular.model_based_algorithms import policy_iteration, value_iteration
from rl.environment.gridworld.gridworld_environment import GridWorld

grid = [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '£'],
        ['#', '.', '.', '$']]

env = GridWorld(grid, 30)


def print_values(policy, value, string):
    print(string)
    env.render(policy, value)
    print()
    print()


print_values(*value_iteration(env, 0.9, 0, 10), 'Value Iteration')
print_values(*policy_iteration(env, 0.9, 0, 10), 'Policy Iteration')
