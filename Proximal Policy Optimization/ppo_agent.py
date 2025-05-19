import numpy as np
from torch import nn
import torch
from torch.distributions import Normal

device = torch.device("cude" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)  # mean
        self.fc_std = nn.Linear(hidden_dim, action_dim)  # variance
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Pendulum [-2, 2]
        self.softplus = nn.Softplus()

    def forward(self, state):
        x = self.relu(self.fc1(state))  # 修正：直接传递 state
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * 2  # 缩放到 Pendulum 的动作范围 [-2, 2]
        std = self.softplus(self.fc_std(x)) + 1e-3  # 确保标准差 > 0
        return mean, std

    def select_action(self, state):
        with torch.no_grad():
            mu, sigma = self.forward(state)
            normal_dist = Normal(mu, sigma)  # assume Gaussian Distribution
            action = normal_dist.sample()
            action = action.clamp(-2,2)
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)

        return value

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
        batch_start_index = np.arange(0, num_states, self.batch_size) #
        memory_indices = np.arange(num_states, dtype=np.int32)
        np.random.shuffle(memory_indices)
        batches = [memory_indices[i:i+self.batch_size] for i in batch_start_index]

        return np.array(self.state_cap), np.array(self.action_cap), np.array(self.reward_cap), np.array(self.value_cap), np.array(self.done_cap), batches

    def clear(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []


class PPO_Agent:
    def __init__(self, state_dim, action_dim, batch_size):
        self.Lr_actor = 3e-4
        self.Lr_critic = 3e-4
        self.gamma = 0.99
        self.lamda = 0.95
        self.Num_epoch = 10
        self.epsilon_clip = 0.2

        self.batch_size = batch_size

        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.Lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.Lr_critic)
        self.replay_memory = ReplayMemory(batch_size)

    def get_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device) # add dim
        action = self.actor.select_action(state)
        value = self.critic.forward(state)
        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]

    def update(self):
        self.old_actor.load_state_dict(self.actor.state_dict()) # make new actor to old then make new actor
        for epoch_i in range(self.Num_epoch):
            memo_states, memo_actions , memo_rewards, memo_values, memo_done, batches = self.replay_memory.sample()
            T = len(memo_rewards)
            memo_advantage = np.zeros(T, dtype=np.float32)

            for t in range(T):
                discout = 1
                a_t = 0
                for k in range(t, T-1):
                    a_t += memo_rewards[k] + self.gamma * memo_values[k+1] * (1- int(memo_done[k])) - memo_values[k]
                    discout *= self.gamma * self.lamda
                memo_advantage[t] = a_t

            with torch.no_grad():
                memo_advantage_tensor = torch.tensor(memo_advantage).unsqueeze(1).to(device)
                memo_values_tensor = torch.tensor(memo_values).to(device)

            memo_stastes_tensor = torch.FloatTensor(memo_states).to(device)
            memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)


            for batch in batches:
                with torch.no_grad():
                    old_mu, old_sigma = self.old_actor(memo_stastes_tensor[batch])
                    old_pi = Normal(old_mu, old_sigma)
                batch_old_probs_tensor = old_pi.log_prob(memo_actions_tensor[batch])

                mu, sigma = self.actor(memo_stastes_tensor[batch])
                pi = Normal(mu, sigma)
                batch_probs_tensor = pi.log_prob(memo_actions_tensor[batch])

                ration = torch.exp(batch_probs_tensor - batch_old_probs_tensor)
                surr1 = ration * memo_advantage_tensor[batch]
                surr2 = torch.clamp(ration, 1- self.epsilon_clip, 1 + self.epsilon_clip) * memo_advantage_tensor[batch]

                actor_loss = -torch.min(surr1,surr2).mean()

                batch_returns = memo_advantage_tensor[batch] + memo_values_tensor[batch]
                batch_old_values = self.critic(memo_stastes_tensor[batch])

                critic_loss = nn.MSELoss()(batch_old_values, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.replay_memory.clear()

    def save_policy(self):
        torch.save(self.actor.state_dict(), "ppo_policy_pendulum_v1.para")











