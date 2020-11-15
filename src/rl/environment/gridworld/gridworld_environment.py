import numpy as np

from rl.environment.environment import Environment


class GridWorld(Environment):
    def __init__(self, grid, max_steps, seed=None):
        """
        :param grid: A matrix that represents the grid world
                grid = [['&', '.', '.', '.'],
                        ['.', '#', '.', '#'],
                        ['.', '.', '.', '£'],
                        ['#', '.', '.', '$']]
                & -> start
                . -> empty path
                # -> obstacle
                £ -> Negative reward (-1)
                $ -> Positive reward (+1)
        :param max_steps: Maximum number of steps allowed
        :param seed: seed value for random number generator
        """

        self.world = np.array(grid)
        self.rows, self.columns = self.world.shape

        n_actions = 4
        n_states = self.world.size + 1

        self.actions = ((-1, 0), (1, 0), (0, -1), (0, 1))

        self.absorbing_state = n_states - 1

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.world.reshape(-1) == '&')[0]] = 1.0

        super(GridWorld, self).__init__(n_states, n_actions, max_steps, pi, seed)

    def position_to_index(self, x, y):
        """
            Converts position(row index & column index) in the grid to index in the flat representation of the grid.
            Formula: (number of columns * row index) + column index

            Example: 2D array: [[1, 2],
                                [3, 4]]
                     Flat array: [1, 2, 3, 4]
                     position of 3 is (1, 0), index of will be ((2 * 1) + 0) = 2

        :param x: row index of the grid
        :param y: column index of the grid
        :return: index in the flat representation of the grid
        """

        return (self.columns * x) + y

    def index_to_position(self, val):
        """
            Converts index in the flat representation of the grid to position(row index & column index) in the grid.
            Formula: row index = (index / number of columns)
                     column index = (index % number of columns)

            Example: Flat array: [1, 2, 3, 4]
                     2D array: [[1, 2],
                                [3, 4]]
                     index of 3 is 2, position of 3 will be ((2 / 2), (2 % 2)) = (1, 0)

        :param val: index in the flat representation of the grid
        :return: row index, column index as a tuple
        """

        return int(val / self.columns), val % self.columns

    def p(self, next_state, state, action):
        """
            Method to calculate probability of transitioning between state and next_state with action
            Algorithm:
                1. calculate row and column indices for state
                2. calculate row and column indices for taking action from state
                3. If the current state is an obstacle or an absorbing_state, then
                    - return 1 if next_state is equal to state, else 0
                4. If the state is a goal state, then
                    - return 1 if next_state is an absorbing_state, else 0
                5. If the next state is not a wall or an obstacle, then
                    - return 1 if next_state is equal to the state calculated in step 2, else 0
                6. If the next state is a wall or an obstacle, then
                    - return 1 if next_state is equal to state, else 0

        :param next_state: Index of next state
        :param state: Index of current state
        :param action: Action to be taken
        :return: Probability of transitioning between state and next_state with action
        """

        x, y = self.index_to_position(state)
        next_x, next_y = x + self.actions[action][0], y + self.actions[action][1]

        if state == self.absorbing_state or self.world[x, y] == '#':
            return int(state == next_state)
        elif self.world[x, y] in ('£', '$'):
            return int(next_state == self.absorbing_state)
        elif 0 <= next_x < self.rows and 0 <= next_y < self.columns and self.world[next_x, next_y] != '#':
            return int(next_state == self.position_to_index(next_x, next_y))
        else:
            return int(state == next_state)

    def r(self, next_state, state, action):
        """
            Method to calculate reward of transitioning between state and next_state with action
            Algorithm:
                1. If the probability of transitioning between state and next state with action is 0, then return 0
                2. If the state or next_state is an absorbing_state, then return 0
                3. else if the value of the next_state is '$'(positive reward goal state), then return +1
                4. else if the value of the next_state is '£'(negative reward goal state), then return -1
                5. else return 0

        :param next_state: Index of next state
        :param state: Index of current state
        :param action: Action to be taken
        :return: Reward for transitioning between state and next_state with action
        """

        if self.p(next_state, state, action) == 0 or self.absorbing_state in (state, next_state):
            return 0
        else:
            token = self.world[self.index_to_position(next_state)]
            if token == '$':
                return +1
            elif token == '£':
                return -1
            else:
                return 0

    def step(self, action):
        """
            Method to take a step for choosing action from current state

        :param action: Action to be taken
        :return: next state, reward, done as a tuple for taking action
        """

        state, reward, done = super(GridWorld, self).step(action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def render(self, policy=None, value=None):
        """
            Method to visualize the GridWorld
            Algorithm:
                1. Calls print_world() to print the GridWorld
                2. If policy is provided, prints policy and value

        :param policy: policy to be rendered
        :param value: value to be rendered
        :return: None
        """

        self.print_world()

        if policy is not None:
            actions = ['u', 'd', 'l', 'r']

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.world.shape))

            print('Value:')
            with self._printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.world.shape))

    def print_world(self):
        """
            Method to print the GridWorld
        :return: None
        """

        print('GridWorld:')
        world = self.world.astype('object')

        if self.state < self.absorbing_state:
            world[self.index_to_position(self.state)] = '@@'

        world[(world == '.') | (world == '&')] = '__'
        world[world == '#'] = '##'
        world[world == '$'] = '+1'
        world[world == '£'] = '-1'

        print(world)
