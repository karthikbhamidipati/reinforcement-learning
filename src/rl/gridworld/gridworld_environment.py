import numpy as np

from rl.environment.environment import Environment


class GridWorld(Environment):
    def __init__(self, rows, columns, max_steps, dist=None, seed=None):
        self.rows = rows
        self.columns = columns

        n_actions = 4
        n_states = rows * columns

        super(GridWorld, self).__init__(n_states, n_actions, max_steps, dist, seed)

        self.rewards = np.zeros((rows, columns), dtype=np.float)
        self.rewards[rows - 1][columns - 1] = 1
        self.rewards[rows - 2][columns - 1] = -1
        self.rewards[1][1] = np.NaN

        self.max_reward = 1
        self.min_reward = -1

        self.actions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def position_to_index(self, x, y):
        return (self.columns * x) + y

    def index_to_position(self, val):
        return int(val / self.columns), val % self.columns

    def p(self, next_state, state, action):
        x, y = self.index_to_position(state)

        next_x, next_y = x + self.actions[action][0], y + self.actions[action][1]

        # Current state is obstacle
        if np.isnan(self.rewards[x, y]):
            return int(state == next_state)
        # Current state is a goal state
        elif ((x == self.rows - 1) or (x == self.rows - 2)) and (y == self.columns - 1):
            return int(state == next_state)
        # next state is a valid state (i.e., shouldn't be a wall or an obstacle)
        elif 0 <= next_x < self.rows and 0 <= next_y < self.columns and not np.isnan(self.rewards[next_x, next_y]):
            return int(next_state == self.position_to_index(next_x, next_y))
        # if next state is wall or an obstacle
        else:
            return int(next_state == state)

    def r(self, next_state, state, action):
        x, y = self.index_to_position(next_state)
        return self.p(next_state, state, action) * np.nan_to_num(self.rewards[x, y])

    def render(self, curr_index):
        for row_idx in range(len(self.rewards)):
            out = ' '
            for col_idx in range(len(self.rewards[row_idx])):
                val = self.rewards[row_idx][col_idx]
                index = self.position_to_index(row_idx, col_idx)
                if index == curr_index:
                    out += 'At '
                elif val == float(0):
                    out += '__ '
                elif val == float(1):
                    out += '+1 '
                elif val == float(-1):
                    out += '-1 '
                else:
                    out += '^^ '
            print(out)
