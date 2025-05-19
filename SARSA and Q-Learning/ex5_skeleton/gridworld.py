"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np
from copy import deepcopy

import copy
from typing import Any, Tuple, Dict, Optional


def clamp(v, minimal_value, maximal_value):
    return min(max(v, minimal_value), maximal_value)


class GridWorldEnv(gym.Env):
    """
    Description:
        An agent moves inside of a small maze.
    Source:
        This environment is a variation on common grid world environments.
    Observation:
        Type: Box(2)
        Num     Observation            Min                     Max
        0       y position             0                       self.map.shape[0] - 1
        1       x position             0                       self.map.shape[1] - 1
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
        The agent is placed into one of the squares marked "s"
    Episode Termination:
        Agent has reached a goal or a trap.
    Solved Requirements:
        
    """

    def __init__(self, map_name='standard'):
        if map_name == 'standard':
            self.map = [
                list("s   "),
                list("    "),
                list("    "),
                list("gt g"),
            ]
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)
            self.start_position = [0, 0]
            self.agent_position = None
            self.trap_reward, self.step_reward = -1.0, 0.0
        elif map_name == 'cliffwalk':
            # TODO 2.1: Implement the CliffWalking environment
            self.map = [
                list("    "),
                list("    "),
                list("    "),
                list("sttg"),
            ]
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)
            self.start_position = [0, 0]
            self.agent_position = None
            self.trap_reward, self.step_reward = -100, -1
        else:
            raise NotImplementedError

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        self.agent_position = deepcopy(self.start_position)
        return self._observe(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)
        match action:
            case 0:  # up
                self.agent_position[0] -= 1
            case 1:  # right
                self.agent_position[1] += 1
            case 2:  # down
                self.agent_position[0] += 1
            case _:  # left
                self.agent_position[1] -= 1

        self.agent_position[0] = clamp(self.agent_position[0], 0, 3)
        self.agent_position[1] = clamp(self.agent_position[1], 0, 3)
        observation = self._observe()
        reward, done = self.step_reward, False
        if 't' == self.map[self.agent_position[0]][self.agent_position[1]]:
            reward, done = self.trap_reward, True
        if 'g' == self.map[self.agent_position[0]][self.agent_position[1]]:
            reward, done = +1, True
        return observation, reward, done, False, {}

    def render(self):
        rendered_map = copy.deepcopy(self.map)
        rendered_map[self.agent_position[0]][self.agent_position[1]] = "A"
        print("--------")
        for row in rendered_map:
            print('|', end='')
            for cell in row:
                print("{}|".format(cell), end='')
            print()
        print("--------")
        return None

    def close(self):
        pass

    def _observe(self):
        return np.array(self.agent_position)
