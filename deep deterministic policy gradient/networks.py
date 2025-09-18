import numpy as np
from torch import nn
import torch

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim=64):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(state_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        x = torch.tanh(self.f3(x)) * 2 # [-2,2] action space: Box(-2.0, 2.0, shape=(1,), dtype=np.float32)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.f1 = nn.Linear(state_dim + action_dim, hidden_dim) # Q(state,action) --> input is state and action, output is Q value
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, 1) # output is Q value scalar

    def forward(self, x, a):
        x = torch.cat([x,a], 1) # concatenate tensor tate,action
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))  # [-2,2]
        return self.f3(x)
