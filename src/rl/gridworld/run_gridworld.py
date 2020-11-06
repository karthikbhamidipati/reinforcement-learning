from rl.gridworld.gridworld_environment import GridWorld

actions = ('8', '2', '4', '6')

prob_dist = [0] * 12
prob_dist[0] = 1

env = GridWorld(3, 4, 30, dist=prob_dist)
start = env.reset()
env.render(start)

done = False

while not done:
    c = input('Pick one possible direction : {}'.format(actions))
    if c not in actions:
        raise Exception('Invalid Action')

    curr_state, r, done = env.step(actions.index(c))
    env.render(curr_state)