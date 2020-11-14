from rl.environment.gridworld.gridworld_environment import GridWorld

actions = ('8', '2', '4', '6')

grid = [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '£'],
        ['#', '.', '.', '$']]

env = GridWorld(grid, 30)
start = env.reset()
env.render()

done = False

while not done:
    c = input('Pick one possible direction : {}'.format(actions))
    if c not in actions:
        raise Exception('Invalid Action')

    curr_state, r, done = env.step(actions.index(c))
    env.render()