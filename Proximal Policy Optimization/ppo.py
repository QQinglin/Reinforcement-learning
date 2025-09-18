import numpy as np
from torch import nn
import torch
from networks import ActorNetwork, CriticNetwork
from torch.distributions import Normal
from transition_memory import ReplayMemory
import time
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class PPO:
    def __init__(self, env, batch_size=64, gamma=0.99, lamb=0.95, lr_actor=0.0001, lr_critic=0.0001):
        self.lamb = lamb
        self.gamma = gamma
        self.Num_epoch = 10
        self.epsilon_clip = 0.2
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.batch_size = batch_size
        self.env = env

        self.actor = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        self.old_actor = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        self.critic = CriticNetwork(self.obs_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.replay_memory = ReplayMemory(batch_size)

    def learn(self,Num_episodes,Num_steps_per_episode):
        # Directory for saving models
        current_path = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(current_path, 'models')
        os.makedirs(model_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        Best_reward = -float('inf')
        total_timesteps = 0
        update_interval = 50
        Reward_buffer = np.zeros(Num_episodes)
        start_time = time.time()

        for episode_i in range(Num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_start_time = time.time()

            for step in range(Num_steps_per_episode):
                action, value = self.get_action(state)
                state_, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                done = truncated or (step + 1 == Num_steps_per_episode)
                self.replay_memory.push(state, action, reward, value, done)
                state = state_
                total_timesteps += 1

                if (step + 1) % update_interval == 0 or done:
                    self.update()

                if done:
                    break

            Reward_buffer[episode_i] = episode_reward
            if episode_reward > Best_reward:
                Best_reward = episode_reward
                torch.save(self.actor.state_dict(), os.path.join(model_dir, f'ppo_actor_{timestamp}.pth'))
                torch.save(self.critic.state_dict(), os.path.join(model_dir, f'ppo_critic_{timestamp}.pth'))

            # Formatted output
            iteration_time = time.time() - episode_start_time
            total_time = time.time() - start_time
            avg_reward = np.mean(
                Reward_buffer[max(0, episode_i - 99):episode_i + 1]) if episode_i > 0 else episode_reward

            print(f"Learning Iteration {episode_i + 1}/{Num_episodes}")
            print(f"Episode Reward: {episode_reward:.2f}")
            print(f"Average Reward (last 100): {avg_reward:.2f}")
            print(f"Best Reward: {Best_reward:.2f}")
            print(f"Total Timesteps: {total_timesteps}")
            print(f"Iteration Time: {iteration_time:.2f}s")
            print(f"Total Time: {total_time:.2f}s")
            print("------------------------------------------------------------------")

        self.env.close()

    def get_action(self,state):
        # add dim from state = [cos(θ), sin(θ), θ̇]to state = [[cos(θ), sin(θ), θ̇]]
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor.select_action(state)  # 假设输出 [1, action_dim]
        value = self.critic(state).squeeze(1)  # 假设输出 [1, 1]
        # 返回时保持形状和类型一致
        return action.detach().cpu().numpy()[0], value.item()

    def compute_returns(self, rewards, values, dones):
        td0 = 0
        next_advantage = 0
        num_transitions = len(rewards)
        memo_advantage = np.zeros(num_transitions, dtype=np.float32)
        returns = np.zeros_like(rewards)
        for step in reversed(range(num_transitions)):
            if step == num_transitions - 1:
                next_value = 0
            else:
                next_value = values[step + 1]
            next_is_terminal = int(dones[step])
            td0 = rewards[step] + self.gamma * (1 - next_is_terminal) * next_value
            delta = td0 - values[step]
            memo_advantage[step] = delta + self.gamma * self.lamb * (1 - next_is_terminal)  * next_advantage
            next_advantage = memo_advantage[step]
            returns[step] = memo_advantage[step] + values[step]
        memo_advantage = (memo_advantage - memo_advantage.mean()) / (memo_advantage.std() + 1e-8)
        return memo_advantage, returns

    def update(self):
        self.old_actor.load_state_dict(self.actor.state_dict())
        for _ in range(self.Num_epoch):
            memo_states, memo_actions, memo_rewards, memo_values, memo_done, batches = self.replay_memory.sample()
            memo_advantage, memo_returns = self.compute_returns(memo_rewards, memo_values, memo_done)
            # x=torch.tensor([1,2,3,4])-->type=(4); x.unsqueeze(0) =  [[1,2,3,4]] --> type=[1,4]; x.unsqueeze(1)=[[1],[2],[3],[4]] --> type=[4,1]
            memo_advantage_tensor = torch.FloatTensor(memo_advantage).unsqueeze(1).to(device) 
            memo_returns_tensor = torch.FloatTensor(memo_returns).unsqueeze(1).to(device)
            memo_states_tensor = torch.FloatTensor(memo_states).to(device)
            memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)

            for batch in batches:
                with torch.no_grad():
                    old_mu, old_sigma = self.old_actor(memo_states_tensor[batch])
                    old_pi = Normal(old_mu, old_sigma)
                    batch_old_probs_tensor = old_pi.log_prob(memo_actions_tensor[batch]).sum(dim=1, keepdim=True) # 行相加

                mu, sigma = self.actor(memo_states_tensor[batch])
                pi = Normal(mu, sigma)
                batch_probs_tensor = pi.log_prob(memo_actions_tensor[batch]).sum(dim=1, keepdim=True)

                ratio = torch.exp(batch_probs_tensor - batch_old_probs_tensor)
                surr1 = ratio * memo_advantage_tensor[batch]
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * memo_advantage_tensor[batch]
                actor_loss = -torch.min(surr1, surr2).mean()

                batch_values = self.critic(memo_states_tensor[batch])
                critic_loss = nn.MSELoss()(batch_values, memo_returns_tensor[batch])

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        self.replay_memory.clear()

    def save_policy(self, path="ppo_policy_pendulum_v1.pth"):
        torch.save(self.actor.state_dict(), path)