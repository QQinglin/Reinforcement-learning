import os.path
import time
import random
import gymnasium as gym
import numpy as np
import torch

from agent_ddpg import DDPGAgent

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = DDPGAgent(state_dim, action_dim) # TODO

# Hyperparameters
Num_episodes = 100
Num_steps = 200
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 10000

reward_buffer = np.empty(Num_episodes)

for episode_i in range(Num_episodes):
    state, others = env.reset()
    episode_reward = 0
    for step_i in range(Num_steps):
        epsilon = np.interp(x=episode_i*Num_episodes+step_i,xp=[0,epsilon_decay],fp=[epsilon_start,epsilon_end])
        random_sample = random.random()
        if random_sample < epsilon:
            action = np.random.uniform(low=-2,high=2,size=action_dim)
        else:
            action = agent.get_action(state)

        state_, reward, done, truncation, info = env.step(action)

        agent.replay_buffer.push(state, action, reward, state_, done)
        state = state_
        episode_reward += reward

        agent.update() # TODO

        if done:
            break
    reward_buffer[episode_i] = episode_i
    print(f"Episode: {episode_i}, Reward: {round(episode_reward,2)}")

current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
timestamp = time.strftime("%Y%m%d-%H%M%S")

torch.save(agent.actor.state_dict(), model+f"ddpg_actor_{timestamp}.path")
torch.save(agent.critic.state_dict(), model+f"ddpg_critic_{timestamp}.path")


env.close()



