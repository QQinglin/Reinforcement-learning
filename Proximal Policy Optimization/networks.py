import numpy as np
from torch import nn
import torch
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)  # mean
        self.fc_std = nn.Linear(hidden_dim, action_dim)  # variance
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Pendulum [-2, 2]
        self.softplus = nn.Softplus()

    def forward(self, state):
        """
        Pendulum-v1 model is continuous action space, Actor network outputs high dimensional Gaussian distribution.
        factorized Gaussian policy : action dimension = n,we assume independent Gaussian distributions for each dimension.
                                    This means the joint distribution over actions is a product of n independent Gaussians
                                     — one for each action dimension.
        :param state:
        :return:
        """
        x = self.relu(self.fc1(state))  #
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * 2  # 缩放到 Pendulum 的动作范围 [-2, 2]
        std = torch.clamp(self.softplus(self.fc_std(x)), min=1e-3, max=2.0)  # softplus 激活函数+ epsilon 确保标准差 > 0
        # for Pendulum-v1 action space has 1 dimension, so output is mean and variance of one Gaussian Distribution
        
        return mean, std

    def select_action(self, state):
        with torch.no_grad():
            mu, sigma = self.forward(state)
            normal_dist = Normal(mu, sigma)  # assume Gaussian Distribution
            action = normal_dist.sample()
            action = action.clamp(-2,2) # action represents the torque , action ->[-2,2] continuous action space

        return action


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)

        return value