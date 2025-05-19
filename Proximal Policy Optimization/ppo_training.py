import os

import time
from ppo_agent import PPO_Agent

import gymnasium as gym
import numpy as np
import torch

scenario = 'Pendulum-v1'
env = gym.make(scenario)
Num_episodes = 10000
Num_steps_per_episode = 1000

# Directory for saving models
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
timestamp = time.strftime("%Y%m%d-%H%M%S")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
batch_size = 25
update_interval = 50
agent = PPO_Agent(state_dim, action_dim,batch_size) # TODO

Reward_buffer = np.empty(shape=Num_episodes)

Best_reward = -2000

for episode_i in range(Num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0

    for step in range(Num_steps_per_episode):
        action, value = agent.get_action(state) # TODO
        state_, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if step + 1 == Num_steps_per_episode:
            done = True
        else:
            done = False
        agent.replay_memory.push(state, action, reward, value, terminated)
        state = state_

        if (step + 1) % update_interval == 0 or (step + 1) == Num_steps_per_episode:
            agent.update() # TODO

    if episode_reward >= -100 and episode_reward > Best_reward:
        agent.save_policy()
        best_reward = episode_reward
        torch.save(agent.actor.state_dict(), model + f'ppo_actor_{timestamp}.pth')
        print(f"Best reward: {best_reward}")

    Reward_buffer[episode_i] = episode_reward
    print(f"Episode {episode_i} reward: {round(episode_reward,2)}")

env.close()