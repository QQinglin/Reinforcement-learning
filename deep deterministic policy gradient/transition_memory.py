import numpy as np
from collections import deque
import random

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, state_, done):
        state = np.expand_dims(state,0) # Pendulum-v1，state [cos(θ), sin(θ), dθ/dt] (3,) --> (1,3) [[cos(θ), sin(θ), dθ/dt]], add batch dimension for network input
        state_ = np.expand_dims(state_,0) # np.expand_dims(state, 0) ，shape = (1, 3)
        self.buffer.append((state, action, reward, state_, done)) # add from left to right. Element is a tuple with 5 elements

    def sample(self, batch_size):
        # 列表装元组：random.sample(self.buffer, batch_size)-> [(s1, a1, r1, s1_, d1), (s2, a2, r2, s2_, d2), (s3, a3, r3, s3_, d3)]
        # *解引用把列表去掉只剩元组：*random.sample(self.buffer, batch_size) -> zip((s1, a1, r1, s1_, d1), (s2, a2, r2, s2_, d2), (s3, a3, r3, s3_, d3))
        # zip 把元组对应位置组成元组再用列表装[(s1, s2, s3), (a1, a2, a3), (r1, r2, r3), (s1_, s2_, s3_), (d1, d2, d3)]
        state, action, reward, state_, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(state_), done

    def __len__(self):
        return len(self.buffer)