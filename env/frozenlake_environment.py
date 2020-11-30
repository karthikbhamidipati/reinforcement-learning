import numpy as np

from env.env_helper import index_to_position, position_to_index
from env.environment import Environment


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
            Constructor for the Frozen lake environment that inherits the class Environment
            1. initialization - create an array of size of the lake
                              - create rows and columns with the shape of lake, e.g. 4*4 or 8*8, etc.
                              - assign a tuple of (x,y) tuples ((-1, 0), (1, 0), (0, -1), (0, 1)) to account for
                                movement in directions            left  , right,  down   , up
                              - allocate number of states (n_states) with size of lake + 1, to account for the absorbing state
                              - allocate number of actions (n_actions) with the length of number of actions
                              - allocate actual number of states to absorbing state, i.e, n_states - 1
                              - create 1D array representing the probability distribution of size of number of states,
                                and then allocate starting position(&) with a probability of 1
                              - create a 3D array to store the transitional probabilities of moving from the current state
                                to the next state considering all the probable actions of up, down, left, right
                                The size of this array will be n_states * n_actions * n_states
                                To explain further, for the current state 0:
                            [[0.95  0.025 0.    0.    0.025 0.    0.    0.    0.    0.    0.    0.  0.    0.    0.    0.    0.]
                             [0.05  0.025 0.    0.    0.925 0.    0.    0.    0.    0.    0.    0.  0.    0.    0.    0.    0.]
                             [0.95  0.025 0.    0.    0.025 0.    0.    0.    0.    0.    0.    0.  0.    0.    0.    0.    0.]
                             [0.05  0.925 0.    0.    0.025 0.    0.    0.    0.    0.    0.    0.  0.    0.    0.    0.    0.]]
                              - call the function_populate_probabilities() to load the precomputed probabilities in the 3D array

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

        super(FrozenLake, self).__init__(n_states, n_actions, max_steps, pi, seed)

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

        state, reward, done = super(FrozenLake, self).step(action)

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
            actions = ['↑', '↓', '←', '→']

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with self._printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

    def _populate_probabilities(self):
        """
            Method to calculate probability of transitioning between state and next_state with action
            Algorithm:
                1. for each state do the steps below
                2. calculate row and column indices for the state
                3. if the state is an absorbing state or a hole or goal, then probability of transitioning to the absorbing state is 1
                   and go back to step 1
                4. for every possible action (up, down, left, right), go to step 5
                5. for slippage in each action
                                - assign the current state to the next state
                                - allocate the x,y coordinate of the next state by adding the movements to the direction,

                                                        add (0 to the x coordinate, +1 to y coordinate)
                                                                             ↑
                    add (-1 to the x coordinate, 0 to y coordinate) ← current_state → add (+1 to the x coordinate, 0 to y coordinate)
                                                                             ↓
                                                        add (0 to the x coordinate, -1 to y coordinate)
                                - if the next state is within the grid, then make the next move, else stay where you are
                                - assign uniform probability distribution of slippage for transitioning from current state to next state with the slippage action
                                 (example, 0.025 (0.1/4) in this scenario)
                                - if the action and slip action are same, then it means the position has not changed, hence the probability is 1-slip,
                                 (example, if slippage is 0.1, then no action from current position results in probability of 0.9)
                                 Assigning higher probability to continue to move in the same direction
        :return: None
        """

        for state in range(self.n_states):
            x, y = index_to_position(state, self.columns)

            if state == self.absorbing_state or self.lake[x, y] in ('#', '$'):
                self._p[state, self.absorbing_state, :] = 1
                continue

            for action in range(self.n_actions):
                for slip_action in range(self.n_actions):
                    next_state = state
                    next_x, next_y = x + self.actions[slip_action][0], y + self.actions[slip_action][1]
                    if 0 <= next_x < self.rows and 0 <= next_y < self.columns:
                        next_state = position_to_index(next_x, next_y, self.columns)

                    self._p[state, next_state, action] += self.slip / self.n_actions
                    if action == slip_action:
                        self._p[state, next_state, action] += 1 - self.slip

    def _populate_rewards(self):
        """
            Method to calculate reward of transitioning between state and next_state with action
            Algorithm:
                1. for each state do the steps below
                2. if the state is a goal state, set the reward for state to absorbing_state for all actions as 1.

        :return: None
        """

        for state in range(self.n_states):
            if state != self.absorbing_state and self.lake[index_to_position(state, self.columns)] == '$':
                self._r[state, self.absorbing_state, :] = 1