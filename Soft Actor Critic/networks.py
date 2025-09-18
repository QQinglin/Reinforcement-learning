import numpy as np
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        self.fc1 = nn.Linear(state_dim + action_dim, self.fc1_dim) # Q(s,a), input->(s,a)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.q = nn.Linear(self.fc2_dim, 1) # output -> q

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1))) # input->(s,a)
        x = F.relu(self.fc2(x))
        q =self.q(x)
        return q

class ValueNetwork(nn.Module):
    def __init__(self, beta, state_dim, fc1_dim, fc2_dim):
        super(ValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        self.fc1 = nn.Linear(state_dim, self.fc1_dim)  # Q(s,a), input->(s,a)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.v = nn.Linear(self.fc2_dim, 1)  # output -> q

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state):
        x = F.relu(self.fc1(state))  # input->(s,a)
        x = F.relu(self.fc2(x))
        v = self.v(x)
        return v

class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.max_action = max_action
        self.alpha = alpha

        self.fc1 = nn.Linear(state_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)

        self.mu = nn.Linear(self.fc2_dim, self.action_dim)
        self.sigma = nn.Linear(self.fc2_dim, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

        self.tiny_positive = 1e-6

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu(x)) * self.max_action # a~[-2,2]
        sigma = self.sigma(x)
        sigma = F.softplus(sigma) + self.tiny_positive
        sigma = torch.clamp(sigma, self.tiny_positive, max=1.0)

        return mu, sigma # same as PPO, output is parameter of Gaussian distribution of action

    def sample_normal(self, state, reparameterize):
        """
        normal sample : a~N(mu.sigma^2), non-differentiable sampling process
        reparameterizing trick : a = mu + sigma*Epsilon, where Epsilon~N(0,1) differentiable sampling process so mu and sigma can propagate in training

        :param state:
        :param reparameterize:
        :return:
        """
        mu, sigma = self.forward(state)
        probability = Normal(mu, sigma) # create a Gaussian Distribution objection
        if reparameterize:
            raw_action = probability.rsample()
        else:
            raw_action = probability.sample()

        tanh_action = torch.tanh(raw_action) # mapping from [-inf, inf] -->[-1,1]

        scaled_action = tanh_action * self.max_action # maping to action scale of the environment: Pendulum-v1 的 [-2, 2]
        log_prob = probability.log_prob(raw_action)    # log(mu(u|s))
        # log p(a) = log p(z) - log |da/dz|，其中 da/dz = 1 - tanh(z)^2
        log_prob -= torch.log(1 - tanh_action.pow(2) + self.tiny_positive)

        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(0) # (1, action_dim)
        log_prob = log_prob.sum(1, keepdim=True) # sum in 1st dimension and keep original dimension

        return scaled_action, log_prob