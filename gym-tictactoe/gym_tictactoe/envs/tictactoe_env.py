import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import random


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Define observation & action spaces
        # Each cell can be 0(empty), 1(X), 2(O)
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=np.int8)
        self.action_space = spaces.Discrete(9)

        self.state = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # 1 = X, 2 = O
        self.done = False

    def hash(self):
        return "".join(str(x) for x in self.state.flatten())

    def available_actions(self):
        return [i for i, x in enumerate(self.state.flatten()) if x == 0]

    def check_done(self):
        # check rows, cols, diags
        for i in range(3):
            if np.all(self.state[i, :] == self.current_player):
                return True, (10 if self.current_player == 1 else -10)
            if np.all(self.state[:, i] == self.current_player):
                return True, (10 if self.current_player == 1 else -10)
        if np.all(np.diag(self.state) == self.current_player) or np.all(
            np.diag(np.fliplr(self.state)) == self.current_player
        ):
            return True, (10 if self.current_player == 1 else -10)

        if not np.any(self.state == 0):
            return True, 0  # draw
        return False, 0

    def step(self, action):
        # Gymnasium step: returns obs, reward, terminated, truncated, info
        if self.done:
            raise ValueError("Game already done — call reset().")

        if self.state[action // 3, action % 3] != 0:
            # Invalid move penalty
            return self.state, -5, True, False, {}

        self.state[action // 3, action % 3] = self.current_player
        done, reward = self.check_done()

        terminated = done
        truncated = False
        info = {}

        if not done:
            # switch player
            self.current_player = 1 if self.current_player == 2 else 2

        self.done = done
        return np.copy(self.state), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        self.done = False
        return np.copy(self.state), {}

    def render(self):
        print("Board:")
        symbols = {0: "-", 1: "X", 2: "O"}
        for row in self.state:
            print(" ".join(symbols[c] for c in row))
