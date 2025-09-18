import numpy as np
from torch import nn
import torch
from networks import Actor, Critic
from transition_memory import ReplayMemory
import time
import os
import torch.optim as optim
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class DDPG:
    def __init__(self, env, gamma, tau, batch_size, lr_actor, lr_critic, memory_capacity):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.gamma = gamma 
        self.tau = tau
        
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict()) # theta^mu' <-- theta^mu
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr_actor)

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict()) # theta^Q' <-- theta^Q
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr_critic)

        self.replay_buffer = ReplayMemory(memory_capacity)

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
        
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = (Num_episodes * Num_steps_per_episode)/2

        for episode_i in range(Num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_start_time = time.time()
            
            for step_i in range(Num_steps_per_episode):
                # exploration before first 10000 steps, after that Epsilon = 0.01 just select action with max Q
                epsilon = np.interp(x=episode_i*Num_steps_per_episode+step_i,xp=[0,epsilon_decay],fp=[epsilon_start,epsilon_end])
                random_sample = random.random()
                if random_sample < epsilon:
                    action = np.random.uniform(low=-2,high=2,size=self.action_dim) # in range [2,-2],randomly uniformly choose number
                else:
                    action = self.get_action(state)

                state_, reward, done, truncation, info = self.env.step(action)

                self.replay_buffer.push(state, action, reward, state_, done)
                state = state_
                episode_reward += reward

                self.update() # TODO

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

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device) # Pendulum-v1，状态 [cos(θ), sin(θ), dθ/dt] (3,) -> (1,3)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, state_s, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.vstack(actions)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device) # add dim to 2
        state_s = torch.FloatTensor(state_s).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Update critic
        next_actions = self.actor_target(state_s) # a' = pi_theta'(s')
        target_Q = self.critic_target(state_s, next_actions.detach()) # Q_beta'(s'_i,pi_theta'(s'))
        target_Q = rewards + (self.gamma * target_Q * (1 - dones))
        y_i = self.critic(states, actions)
        critic_loss = nn.MSELoss()(y_i, target_Q)
        self.critic_optimizer.zero_grad() # clear old grad form the last step
        critic_loss.backward() # compute the derivatives of the loss
        self.critic_optimizer.step()

        # Update actor net
        actor_loss = -self.critic(states, self.actor(states)).mean() # Loss = -Q(s_i,pi_theta(s_i))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks of critic and actor
        for target_param, param, in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param, in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)