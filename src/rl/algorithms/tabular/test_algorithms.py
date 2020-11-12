import numpy as np

from rl.algorithms.tabular.model_based_algorithms import policy_iteration, value_iteration
from rl.environment.gridworld.gridworld_environment import GridWorld

prob_dist = [0] * 12
prob_dist[0] = 1

env = GridWorld(3, 4, 30, dist=prob_dist)

env.render(env.reset())
print()
actions = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


def print_values(policy, value, string):
    print(string)
    print(np.array([actions[a] for a in policy]).reshape(3, 4))
    print(value.reshape(3, 4))
    print()


print_values(*value_iteration(env, 0.99, 0, 10), 'Value Iteration')
print_values(*policy_iteration(env, 0.99, 0, 10), 'Policy Iteration')
