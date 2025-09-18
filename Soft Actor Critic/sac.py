import numpy as np
import torch
from networks import  ValueNetwork,  CriticNetwork, ActorNetwork
from transition_memory import ReplayMemory
import time
import os

import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class SAC:
    def __init__(self, env, alpha, beta, gamma, tau, batch_size, lr_actor, lr_critic, memo_capacity):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.gamma = gamma 
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        
        self.layer1_dim=64 
        self.layer2_dim=64
        
        self.memory = ReplayMemory(capacity=memo_capacity, state_dim=self.state_dim, action_dim=self.action_dim)
        
        self.critic_1 = CriticNetwork(beta=beta, state_dim=self.state_dim, action_dim=self.action_dim, fc1_dim=self.layer1_dim,
                                      fc2_dim=self.layer2_dim).to(device)
        
        self.critic_2 = CriticNetwork(beta=beta, state_dim=self.state_dim, action_dim=self.action_dim, fc1_dim=self.layer1_dim,
                                      fc2_dim=self.layer2_dim).to(device)
        
        self.value = ValueNetwork(beta=beta, state_dim=self.state_dim, fc1_dim=self.layer1_dim,
                                   fc2_dim=self.layer2_dim).to(device)
        
        self.target_value = ValueNetwork(beta=beta, state_dim=self.state_dim, fc1_dim=self.layer1_dim,
                                  fc2_dim=self.layer2_dim).to(device)
        
        self.actor = ActorNetwork(alpha=alpha, state_dim=self.state_dim, action_dim=self.action_dim, fc1_dim=self.layer1_dim,
                                  fc2_dim=self.layer2_dim, max_action=2).to(device)
        
    def learn(self,Num_episodes,Num_steps_per_episode):
        # Directory for saving models
        current_path = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(current_path, 'models')
        os.makedirs(model_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        Best_reward = -float('inf')
        total_timesteps = 0
        Reward_buffer = np.zeros(Num_episodes)
        start_time = time.time()
        
        for episode_i in range(Num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_start_time = time.time()
            
            for step_i in range(Num_steps_per_episode):
                action = self.get_action(state) # TODO
                state_, reward, done, trunc, info = self.env.step(action)
                self.add_memory(state, action, reward, state_, done) # TODO
                episode_reward += reward
                state = state_
                self.update() #TODO

                if done:
                    break
        
            Reward_buffer[episode_i] = episode_reward
            if episode_reward > Best_reward:
                Best_reward = episode_reward
                torch.save(self.actor.state_dict(), os.path.join(model_dir, f'sac_actor_{timestamp}.pth'))
                torch.save(self.critic_1.state_dict(), os.path.join(model_dir, f'sac_critic1_{timestamp}.pth'))
                torch.save(self.critic_2.state_dict(), os.path.join(model_dir, f'sac_critic2_{timestamp}.pth'))

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
        y_i = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, y_i)
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

        actor_loss = self.alpha * log_probs - critic_value
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
