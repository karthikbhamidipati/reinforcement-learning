from env.frozenlake_environment import FrozenLake
from env.gridworld_environment import GridWorld


def run_env(env, actions):
    """
        Method to manually run and visualize the environment

    :param env: environment to run
    :param actions: possible actions
    :return: None
    """

    env.reset()
    env.render()

    done = False

    while not done:
        c = input('Pick one possible direction : {}'.format(actions))
        if c not in actions:
            raise Exception('Invalid Action')

        curr_state, r, done = env.step(actions.index(c))
        env.render()
        print("Reward : {}\n".format(r))


def run_frozenlake():
    """
        Method to manually run and visualize the FrozenLake environment

    :return: None
    """

    actions = ('8', '2', '4', '6')

    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    run_env(FrozenLake(lake, 0, 30), actions)


def run_gridworld():
    """
        Method to manually run and visualize the GridWorld environment

    :return: None
    """

    actions = ('8', '2', '4', '6')

    grid = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '£'],
            ['#', '.', '.', '$']]

    run_env(GridWorld(grid, 30), actions)


run_frozenlake()
