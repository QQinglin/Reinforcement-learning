import numpy as np
from collections import deque
import random

class ReplayMemory:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state_memory = np.zeros((capacity, state_dim))
        self.action_memory = np.zeros((capacity, action_dim))
        self.state__memory = np.zeros((capacity, state_dim))
        self.reward_memory = np.zeros(capacity)
        self.done_memory = np.zeros(capacity)
        self.memo_counter = 0

    def push(self, state, action, reward, state_, done):
        # Overwrite the earliest data in list
        index = self.memo_counter % self.capacity
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.state__memory[index] = state_
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.memo_counter += 1

    def sample(self, batch):
        max_mem = min(self.memo_counter, self.capacity)
        batch = np.random.choice(max_mem, batch, replace=False) ## type: np.array
        batch_state = self.state_memory[batch]
        batch_action = self.action_memory[batch]
        batch_reward = self.reward_memory[batch]
        batch_state_ = self.state__memory[batch]
        batch_done = self.done_memory[batch]

        return batch_state, batch_action, batch_reward, batch_state_, batch_done