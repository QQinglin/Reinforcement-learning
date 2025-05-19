import random

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import torch.optim as optim

# Hyperparameter
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
GAMMA = 0.99
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
TAU = 5e-3

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device typy: ",device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim=64):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(state_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        x = torch.tanh(self.f3(x)) * 2 # [-2,2]
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.f1 = nn.Linear(state_dim + action_dim, hidden_dim) # Q(state,action)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, 1)

    def forward(self, x,a):
        x = torch.cat([x,a], 1)
        x = torch.relu(self.f1(x))
        x = torch.tanh(self.f2(x)) * 2  # [-2,2]
        return self.f3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, state_, done):
        state = np.expand_dims(state,0)  # Pendulum-v1，状态 [cos(θ), sin(θ), dθ/dt] (3,)
        state_ = np.expand_dims(state_,0) # np.expand_dims(state, 0) 后，state 形状变为 (1, 3), 有 1 行 3 列，相当于 [[cos(θ), sin(θ), dθ/dt]]
        self.buffer.append((state, action, reward, state_, done))

    def sample(self, batch_size):
        # random.sample(self.buffer, batch_size)-> [(s1, a1, r1, s1_, d1), (s2, a2, r2, s2_, d2), (s3, a3, r3, s3_, d3)]
        # *random.sample(self.buffer, batch_size) -> zip((s1, a1, r1, s1_, d1), (s2, a2, r2, s2_, d2), (s3, a3, r3, s3_, d3))
        # [(s1, s2, s3), (a1, a2, a3), (r1, r2, r3), (s1_, s2_, s3_), (d1, d2, d3)]
        state, action, reward, state_, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(state_), done

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self,state_dim,action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(),LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), LR_CRITIC)

        self.replay_buffer = ReplayMemory(MEMORY_CAPACITY)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device) # Pendulum-v1，状态 [cos(θ), sin(θ), dθ/dt] (3,) -> (1,3)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, state_s, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.vstack(actions)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        state_s = torch.FloatTensor(state_s).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Update critic
        next_actions = self.actor_target(state_s)
        target_Q = self.critic_target(state_s, next_actions.detach())
        target_Q = rewards + (GAMMA * target_Q * (1 - dones))
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad() # clear old grad form the last step
        self.critic_optimizer.zero_grad()
        critic_loss.backward() # compute the derivatives of the loss
        self.critic_optimizer.step()

        # Update actor net
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks of critic and actor
        for target_param, param, in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)






















