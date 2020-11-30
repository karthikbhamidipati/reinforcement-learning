import numpy as np

from env.env_helper import position_to_index, index_to_position
from env.environment import Environment


class GridWorld(Environment):
    def __init__(self, grid, max_steps, seed=None):
        """
        :param grid: A matrix that represents the grid world
                Example:
                    grid = [['&', '.', '.', '.'],
                            ['.', '#', '.', '#'],
                            ['.', '.', '.', '£'],
                            ['#', '.', '.', '$']]
                    & -> start
                    . -> empty path
                    # -> obstacle
                    £ -> Negative reward (-1)
                    $ -> Positive reward (+1)
        :param max_steps: The maximum number of time steps in an episode
        :param seed: A seed to control the random number generator (optional)
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

        self._p = np.zeros((self.n_states, self.n_states, self.n_actions), dtype=float)
        self._r = np.zeros_like(self._p)

        self._populate_probabilities()
        self._populate_rewards()

    def p(self, next_state, state, action):
        """
            Method to return the probability of transitioning from current state to the next state with action

        :param next_state: Index of next state
        :param state: Index of current state
        :param action: Action to be taken
        :return: Probability of transitioning between state and next_state with action
        """

        return self._p[state, next_state, action]

    def r(self, next_state, state, action):
        """
            Method to return the reward when transitioning from current state to the next state with action

        :param next_state: Index of next state
        :param state: Index of current state
        :param action: Action to be taken
        :return: Reward for transitioning between state and next_state with action
        """

        return self._r[state, next_state, action]

    def step(self, action):
        """
            Method to take a step for choosing action from current state

        :param action: Action to be taken
        :return: next state, reward, done as a tuple for taking action
        """

        state, reward, done = super(GridWorld, self).step(action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def get_prob_rewards(self):
        """
            Method to get the probabilities and rewards for the env.

        :return: probabilities, rewards as numpy arrays
        """

        return self._p, self._r

    def render(self, policy=None, value=None):
        """
            Method to visualize the GridWorld
            Algorithm:
                1. Prints the GridWorld
                2. If policy is provided, prints policy and value

        :param policy: policy to be rendered
        :param value: value to be rendered
        :return: None
        """

        print('FrozenLake:')
        world = self.world.copy()
        if self.state < self.absorbing_state:
            world[index_to_position(self.state, self.columns)] = '@'
        print(world)

        if policy is not None:
            actions = ['↑', '↓', '←', '→']

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.world.shape))

            print('Value:')
            with self._printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.world.shape))

    def _populate_probabilities(self):
        """
            Method to calculate probability of transitioning between state and next_state with action
            Algorithm:
                1. for each state do the steps below
                2. calculate row and column indices for the state
                3. if the state is an absorbing state or a goal state, then probability of transitioning to the absorbing state is 1
                   and go back to step 1
                4. for every possible action (up, down, left, right), go to step 5
                5. find the next state and store the probability of transitioning from state to next state using action as 1.

        :return: None
        """

        for state in range(self.n_states):
            x, y = index_to_position(state, self.columns)

            if state == self.absorbing_state or self.world[x, y] in ('£', '$'):
                self._p[state, self.absorbing_state, :] = 1
                continue

            for action in range(self.n_actions):
                next_state = state
                next_x, next_y = x + self.actions[action][0], y + self.actions[action][1]
                if 0 <= next_x < self.rows and 0 <= next_y < self.columns and self.world[next_x, next_y] != '#' and \
                        self.world[x, y] != '#':
                    next_state = position_to_index(next_x, next_y, self.columns)

                self._p[state, next_state, action] = 1

    def _populate_rewards(self):
        """
            Method to calculate reward of transitioning between state and next_state with action
            Algorithm:
                1. for each state do the steps below
                2. if the state is a goal state, set the reward for state to absorbing_state for all actions as 1.

        :return: None
        """

        for state in range(self.n_states):
            if state != self.absorbing_state:
                if self.world[index_to_position(state, self.columns)] == '$':
                    self._r[state, self.absorbing_state, :] = 1
                elif self.world[index_to_position(state, self.columns)] == '£':
                    self._r[state, self.absorbing_state, :] = -1
