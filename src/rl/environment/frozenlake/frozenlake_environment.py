import numpy as np

from rl.environment.env_helper import index_to_position, position_to_index
from rl.environment.environment import Environment


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        :param lake: A matrix that represents the lake.
                Example:
                    lake =  [['&', '.', '.', '.'],
                             ['.', '#', '.', '#'],
                             ['.', '.', '.', '#'],
                             ['#', '.', '.', '$']]
                    & -> start
                    . -> frozen
                    # -> hole
                    $ -> goal
        :param slip: The probability that the agent will slip
        :param max_steps: The maximum number of time steps in an episode
        :param seed: A seed to control the random number generator (optional)
        """

        self.lake = np.array(lake)
        self.rows, self.columns = self.lake.shape
        self.slip = slip
        self.actions = ((-1, 0), (1, 0), (0, -1), (0, 1))

        n_states = self.lake.size + 1
        n_actions = len(self.actions)

        self.absorbing_state = n_states - 1

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake.reshape(-1) == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        super(FrozenLake, self).__init__(n_states, n_actions, max_steps, pi, seed)

    def p(self, next_state, state, action):
        """
            Method to calculate probability of transitioning between state and next_state with action
            Algorithm:
                1. calculate row and column indices for state
                2. If the state is an absorbing_state or a hole or a goal state
                    - return 1 if next_state is an absorbing_state
                3. Creating an action probability array with
                TODO

        :param next_state: Index of next state
        :param state: Index of current state
        :param action: Action to be taken
        :return: Reward for transitioning between state and next_state with action
        """

        x, y = index_to_position(state, self.columns)

        if state == self.absorbing_state or self.lake[x, y] in ('#', '$'):
            return int(next_state == self.absorbing_state)

        action_prob = np.array([self.slip / 4] * self.n_actions)
        action_prob[action] = 1 - ((self.slip * 3) / 4)
        adjacent_states = []

        for i in range(self.n_actions):
            next_x, next_y = x + self.actions[i][0], y + self.actions[i][1]
            if 0 <= next_x < self.rows and 0 <= next_y < self.columns:
                adjacent_states.append(position_to_index(next_x, next_y, self.columns))
            else:
                adjacent_states.append(state)

        if next_state not in adjacent_states:
            return 0
        else:
            return action_prob[adjacent_states.index(next_state)]

    def r(self, next_state, state, action):
        """
            TODO

        :param next_state: Index of next state
        :param state: Index of current state
        :param action: Action to be taken
        :return: Reward for transitioning between state and next_state with action
        """

        if self.p(next_state, state, action) == 0:
            return 0
        elif self.absorbing_state in (state, next_state):
            return 0
        elif self.lake[index_to_position(next_state, self.columns)] != '$':
            return 0
        else:
            return 1

    def step(self, action):
        """
            Method to take a step for choosing action from current state

        :param action: Action to be taken
        :return: next state, reward, done as a tuple for taking action
        """

        state, reward, done = super(FrozenLake, self).step(action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def render(self, policy=None, value=None):
        """
            Method to visualize the FrozenLake
            Algorithm:
                1. Prints the FrozenLake
                2. If policy is provided, prints policy and value

        :param policy: policy to be rendered
        :param value: value to be rendered
        :return: None
        """

        print('FrozenLake:')
        lake = self.lake.copy()
        if self.state < self.absorbing_state:
            lake[index_to_position(self.state, self.columns)] = '@'
        print(lake)

        if policy is not None:
            actions = ['u', 'd', 'l', 'r']

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with self._printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
