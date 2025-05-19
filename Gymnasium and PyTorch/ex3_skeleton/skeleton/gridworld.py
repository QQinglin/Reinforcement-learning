import copy
from typing import Any, Tuple, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def clamp(v, minimal_value, maximal_value):
    return min(max(v, minimal_value), maximal_value)


class GridWorldEnv(gym.Env):
    """
    This environment is a variation on common grid world environments.
    Observation:
        Type: Box(2)
        Num     Observation            Min                     Max
        0       x position             0                       self.map.shape[0] - 1
        1       y position             0                       self.map.shape[1] - 1
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Go up
        1     Go right
        2     Go down
        3     Go left
        Each action moves a single cell.
    Reward:
        Reward is 0 at non-goal points, 1 at goal positions, and -1 at traps
    Starting State:
        The agent is placed at [0, 0]
    Episode Termination:
        Agent has reached a goal or a trap.
    Solved Requirements:
        
    """

    def __init__(self):
        self.map = [
            list("s   "),
            list("    "),
            list("    "),
            list("gt g"),
        ]

        # TODO: Define your action_space and observation_space here
        super(GridWorldEnv, self).__init__()

        self.grid_size = 4
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low = np.array([0,0]),
                                            high = np.array([self.grid_size - 1,self.grid_size - 1]), dtype=np.int32)
        self.agent_position = None


    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # TODO: Write your implementation here
        self.agent_position = [0,0]

        return self._observe(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = None
        done = None
        # TODO: Write your implementation here
        row, col = self.agent_position

        new_row, new_col = row, col
        if action == 0:
            new_row = max(0, new_row - 1)
        if action == 1:
            new_col = min(self.grid_size - 1, new_col + 1)
        if action == 2:
            new_row = min(self.grid_size - 1, new_row + 1)
        if action == 3:
            new_col = max(0, new_col - 1)

        self.agent_position = [new_row, new_col]

        reward = 0
        done = False
        cell = self.map[new_row][new_col]

        if cell == 'g':
            reward = 1
            done = False
        elif cell == 't':
            reward = -1
            done = True

        observation = self._observe()
        return observation, reward, done, False, {}

    def render(self):
        rendered_map = copy.deepcopy(self.map)
        rendered_map[self.agent_position[0]][self.agent_position[1]] = "A"
        print("--------")
        for row in rendered_map:
            print("".join(["|", " "] + row + [" ", "|"]))
        print("--------")
        return None

    def close(self):
        pass

    def _observe(self):
        return np.array(self.agent_position)
