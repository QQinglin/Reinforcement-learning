import numpy as np

class ReplayMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []

    def push(self, state, action, reward, value, done):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap.append(value)
        self.done_cap.append(done)

    def sample(self):
        num_states = len(self.state_cap)
        batch_start_index = np.arange(0, num_states, self.batch_size) # batch_size = 3 [0,2,4]
        memory_indices = np.arange(num_states, dtype=np.int32) # [0, 1, 2,3,4,5,6]
        np.random.shuffle(memory_indices) # 将索引[0, 1, 2,3,4,5,6] 打乱[2,5,3,1,4,6]
        #batches = [
        #    memory_indices[0:2],  # [2, 5]
        #    memory_indices[2:4],  # [3, 1]
        #    memory_indices[4:6]  # [4, 6]
        #]
        # batches = [[2, 5], [3, 1], [4, 6]]
        batches = [memory_indices[i:i+self.batch_size] for i in batch_start_index] # batch = [memory_indices[0:2]]

        return (np.array(self.state_cap), np.array(self.action_cap), np.array(self.reward_cap),
                np.array(self.value_cap), np.array(self.done_cap), batches)

    def clear(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []