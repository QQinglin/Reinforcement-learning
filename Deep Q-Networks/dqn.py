import sys
import argparse
import random
from random import sample
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from neural_network import NeuralNetwork



class DQN:
    """The DQN method."""

    def __init__(self, env, replay_size=10000, batch_size=32, gamma=0.99, sync_after=5, lr=0.001, use_target_network=False):
        """ Initializes the DQN method.

        Parameters
        ----------
        env: gym.Environment
            The gym environment the agent should learn in.
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.
        sync_after: int
            Timesteps after which the target network should be synchronized with the main network.
        lr: float
            Adam optimizer learning rate.
        use_target_network: bool
            Whether to use target network
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma
        self.use_target_network = use_target_network

        # Initialize DQN network
        self.net = NeuralNetwork(self.obs_dim, self.act_dim)

        if self.use_target_network:
            # TODO 1.6: Initialize DQN target network
            self.target_net = NeuralNetwork(self.obs_dim, self.act_dim)

        # Set up optimizer, only needed for DQN network
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        self.best_episode_data = None
        self.best_mean_reward = -float('inf')

    def learn(self, timesteps):
        """Train the agent for timesteps steps inside self.env.
        After every step taken inside the environment observations, rewards, etc. have to be saved inside the replay buffer.
        If there are enough elements already inside the replay buffer (>batch_size), compute MSBE loss and optimize DQN network.

        Parameters
        ----------
        timesteps: int
            Number of timesteps to optimize the DQN network.
        """
        all_rewards = []
        episode_rewards = []
        episode_data = []
        episode_count = 0
        reward_threshold = 20
        window_size = 10

        s, _ = self.env.reset()
        for timestep in range(1, timesteps + 1):
            sys.stdout.write('\rTimestep: {}/{}'.format(timestep, timesteps))
            sys.stdout.flush()

            epsilon = epsilon_by_timestep(timestep)
            a = self.predict(s, epsilon)

            s_, r, terminated, _, _ = self.env.step(a)
            self.replay_buffer.put(s, a, r, s_, terminated)

            episode_data.append((s.copy(), a))

            s = s_
            episode_rewards.append(r)

           ##
            if terminated  or len(episode_rewards) >= 500:
                episode_count += 1
                total_reward = sum(episode_rewards)
                all_rewards.append(total_reward)

                # 计算最近 window_size 个 episode 的平均奖励
                if len(all_rewards) >= window_size:
                    mean_reward = np.mean(all_rewards[-window_size:])
                    print(
                        f"\nEpisode {episode_count}, Total Reward: {total_reward}, Mean Reward (last {window_size} episodes): {mean_reward:.2f}")

                    # 如果平均奖励超过阈值，保存该 episode 的数据
                    if mean_reward > reward_threshold and mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.best_episode_data = episode_data
                        print(f"Saved best episode at episode {episode_count} with mean reward {mean_reward:.2f}")

                s, _ = self.env.reset()
                episode_rewards = []
                episode_data = []
            ##

            if len(self.replay_buffer) > self.batch_size:
                self.opt.zero_grad()
                loss = self.compute_msbe_loss()
                loss.backward()
                self.opt.step()

            if self.use_target_network and 0 == timestep % self.sync_after:
                # TODO 1.6: Synchronize DQN target network
                self.target_net.load_state_dict(self.net.state_dict()) # TODO
                pass

            if timestep % 500 == 0:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)

    def predict(self, s, epsilon=0.0):
        """Predict the best action based on state. With probability epsilon take random action

        Returns
        -------
        int
            The action to be taken.
        """

        # TODO 1.3: Implement epsilon-greedy action selection
        if random.random() > epsilon:
            s_tensor = torch.as_tensor(s, dtype=torch.float32)
            q_values = self.net.forward(s_tensor.unsqueeze(0)) # (2,) -> (1,2) increase one dim
            max_q_idx = torch.argmax(input=q_values)
            action = max_q_idx.detach().item()
            return action
        else:
            return self.env.action_space.sample()

    def compute_msbe_loss(self):
        """Compute the MSBE loss between self.dqn_net predictions and expected Q-values.

        Returns
        -------
        float
            The MSE between Q-value prediction and expected Q-values.
        """

        # TODO 1.5: Sample from replay buffer
        states, actions, rewards, states_, terminated = [], [], [], [], []
        indexs = self.replay_buffer.get(self.batch_size)

        for idx in indexs:
            batch_s, batch_a, batch_r, batch_s_, batch_terminated = self.replay_buffer.buffer[idx]
            states.append(batch_s)
            actions.append(batch_a)
            rewards.append(batch_r)
            states_.append(batch_s_)
            terminated.append(batch_terminated)

        batch_states_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        batch_actions_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.int64).unsqueeze(-1) # increase one dim (64,) -> (64,1)
        batch_rewards_tensor = torch.as_tensor(np.asarray(rewards), dtype=torch.float32).unsqueeze(-1)
        batch_states__tensor = torch.as_tensor(np.asarray(states_), dtype=torch.float32)
        batch_terminated_tensor = torch.as_tensor(np.asarray(terminated), dtype=torch.float32).unsqueeze(-1)

        # Convert to Tensors and stack for easier processing -> shape (batch_size, state_dimensionality)
        states = torch.stack([torch.Tensor(state) for state in states])
        states_ = torch.stack([torch.Tensor(state_) for state_ in states_])

        # TODO 1.5: Extract Q-values for states
        # Compute Q-values (batch_size x num_actions), select Q-values of actions actually taken (batch_size)
        q_values = self.net.forward(batch_states_tensor)
        a_q_values = torch.gather(input=q_values, dim=1, index=batch_actions_tensor)

        # TODO 1.5: Extract Q-values for states_
        # Compute target either using same or target network (batch_size x num_actions), calculate max (batch_size)
        if self.use_target_network:
            # TODO 1.6: Use target network for updates
            q_values_ = self.target_net.forward(batch_states__tensor).max(dim=1, keepdim=True)[0]
        else:
            q_values_ = self.net.forward(batch_states_tensor).max(dim=1, keepdim=True)[0]

        # TODO 1.5: Compute update target
        # The target we want to update our network towards
        target_q_values = batch_rewards_tensor + self.gamma * (1 - batch_terminated_tensor) * q_values_
        # Calculate loss
        loss = F.mse_loss(a_q_values, target_q_values)
        return loss


def epsilon_by_timestep(timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=10000):
    """Linearly decays epsilon from epsilon_start to epsilon_final in frames_decay timesteps"""

    # TODO 1.2: Implement epsilon decay function  (can be visualized with flag --plot_epsilon)
    epsilon = np.interp(timestep, [0, frames_decay], [epsilon_start, epsilon_final])
    return epsilon

def episode_reward_plot(rewards, timestep, window_size=7, step_size=1):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title(f"Episode Rewards at Timestep {timestep}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()


def visualize_agent(env, agent, num_episodes=5, max_steps=500):
    """
    Visualize the agent's performance after training by running multiple episodes.

    Parameters:
    - env: Gym environment with render_mode='human'
    - agent: Trained DQN agent
    - num_episodes: Number of episodes to visualize
    - max_steps: Maximum steps per episode
    """
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        print(f"\nStarting visualization episode {episode + 1}/{num_episodes}")
        while not done and step < max_steps:
            # 使用训练好的模型选择动作（epsilon=0 表示完全贪婪策略）
            action = agent.predict(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            env.render()
            step += 1
            done = terminated or truncated

            # 稍微减慢动画速度以便观察
            time.sleep(0.02)

        print(f"Episode {episode + 1} finished with reward: {episode_reward}")

    env.close()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_timesteps', '-steps', type=int, default=10000)
    parser.add_argument('--replay_size', '-replay', type=int, default=10000)
    parser.add_argument('--batch_size', '-batch', type=int, default=32)
    parser.add_argument('--gamma', '-gamma', type=float, default=0.99)
    parser.add_argument('--sync_after', '-sync', type=int, default=5)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--use_target_network', '-target', action='store_true')
    parser.add_argument('--plot_epsilon', '-plot_eps', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    _args = parse()

    # Plot epsilon rate over time
    if _args.plot_epsilon:
        plt.plot([epsilon_by_timestep(i) for i in range(_args.n_timesteps)])
        plt.show()

    # Train the DQN agent
    dqn = DQN(gym.make("CartPole-v1"), replay_size=_args.replay_size, batch_size=_args.batch_size, gamma=_args.gamma,
              sync_after=_args.sync_after, lr=_args.learning_rate, use_target_network=_args.use_target_network)
    dqn.learn(_args.n_timesteps)

    # Visualize the agent
    visualize_agent(gym.make("CartPole-v1", render_mode='human'), dqn)
