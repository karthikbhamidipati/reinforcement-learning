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
        c = input('Pick number {} corresponding to possible directions {}:'.format(*actions))
        if c not in actions[0]:
            raise Exception('Invalid Action')

        curr_state, r, done = env.step(actions[0].index(c))
        env.render()
        print("Reward : {}\n".format(r))


def run_small_frozenlake():
    """
        Method to manually run and visualize the Small FrozenLake environment

    :return: None
    """

    actions = (('8', '2', '4', '6'), ('↑', '↓', '←', '→'))

    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    run_env(FrozenLake(lake, 0.1, 16), actions)


def run_big_frozenlake():
    """
        Method to manually run and visualize the Big FrozenLake environment

    :return: None
    """

    actions = (('8', '2', '4', '6'), ('↑', '↓', '←', '→'))

    lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '#', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '#', '#', '.', '.', '.', '#', '.'],
            ['.', '#', '.', '.', '#', '.', '#', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '$']]

    run_env(FrozenLake(lake, 0.1, 64), actions)


def run_gridworld():
    """
        Method to manually run and visualize the GridWorld environment

    :return: None
    """

    actions = (('8', '2', '4', '6'), ('↑', '↓', '←', '→'))

    grid = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '£'],
            ['#', '.', '.', '$']]

    run_env(GridWorld(grid, 16), actions)


# run_gridworld()
run_small_frozenlake()
# run_big_frozenlake()
