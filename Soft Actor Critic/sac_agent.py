import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

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
        index = self.memo_counter % self.capacity
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.state__memory[index] = state_
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.memo_counter += 1

    def sample(self, batch_size):
        current_memo_size = min(self.memo_counter, batch_size)
        batch = np.random.choice(current_memo_size, batch_size,replace=False)
        batch_state = self.state_memory[batch]
        batch_action = self.action_memory[batch]
        batch_reward = self.reward_memory[batch]
        batch_state_ = self.state__memory[batch]
        batch_done = self.done_memory[batch]

        return batch_state, batch_action, batch_reward, batch_state_, batch_done

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

        return mu, sigma

    def sample_normal(self, state, reparameterize):
        mu, sigma = self.forward(state)
        probability = Normal(mu, sigma)
        if reparameterize:
            raw_action = probability.rsample()
        else:
            raw_action = probability.sample()

        tanh_action = torch.tanh(raw_action) # [-inf, inf] -->[-1,1]

        scaled_action = tanh_action * self.max_action
        log_prob = probability.log_prob(raw_action)    # log(mu(u|s))
        log_prob -= torch.log(1 - tanh_action.pow(2) + self.tiny_positive)

        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(0)
        log_prob = log_prob.sum(1, keepdim=True) # sum in 1st dimension and keep original dimension

        return scaled_action, log_prob

class SACAgent:
    def __init__(self, state_dim, action_dim, memo_capacity, alpha, beta, gamma, tau, layer1_dim, layer2_dim, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayMemory(capacity=memo_capacity, state_dim=state_dim, action_dim=action_dim)
        self.critic_1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim, fc1_dim=layer1_dim,
                                      fc2_dim=layer2_dim).to(device).to(device)
        self.critic_2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim, fc1_dim=layer1_dim,
                                      fc2_dim=layer2_dim).to(device).to(device)
        self.value = ValueNetwork(beta=beta, state_dim=state_dim, fc1_dim=layer1_dim,
                                   fc2_dim=layer2_dim).to(device).to(device)
        self.target_value = ValueNetwork(beta=beta, state_dim=state_dim, fc1_dim=layer1_dim,
                                  fc2_dim=layer2_dim).to(device).to(device)
        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim, fc1_dim=layer1_dim,
                                  fc2_dim=layer2_dim, max_action=2).to(device)


    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        action, _ = self.actor.sample_normal(state, reparameterize=False)
        return action.cpu().detach().numpy()

    def add_memory(self, state, action, reward, state_, done):
        self.memory.push(state, action, reward, state_, done)

    def update(self):
        if self.memory.memo_counter < self.batch_size:
            return

        state, action, reward, state_, done = self.memory.sample(self.batch_size)

        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        state_ = torch.tensor(state_, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.bool).to(device)

        value = self.value(state).view(-1)

        with torch.no_grad():
            value_ = self.target_value(state_).view(-1)
            value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Update target_value network
        for target_param, param, in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1- self.tau) * target_param.data)

        # Actor network
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        actions = actions.detach()

        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)

        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Soft Q-function
        with torch.no_grad():
            q_hat = reward + self.gamma * value_

        q1_old_policy = self.critic_1.forward(state, actions).view(-1)
        q2_old_policy = self.critic_2.forward(state, actions).view(-1)

        critic_1_loss = F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = F.mse_loss(q2_old_policy, q_hat)

        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2.optimizer.step()





































