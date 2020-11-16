from rl.environment.frozenlake.frozenlake_environment import FrozenLake

actions = ('8', '2', '4', '6')

lake = [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

env = FrozenLake(lake, 0, 30)
start = env.reset()
env.render()

done = False

while not done:
    c = input('Pick one possible direction : {}'.format(actions))
    if c not in actions:
        raise Exception('Invalid Action')

    curr_state, r, done = env.step(actions.index(c))
    env.render()
