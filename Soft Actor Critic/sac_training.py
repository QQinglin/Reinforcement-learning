import os
import time

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import gymnasium as gym
import numpy as np
import torch
from sac_agent import SACAgent

torch.autograd.set_detect_anomaly(True)


scenario = 'Pendulum-v1'
env = gym.make(id='Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

memory_size = 10000
agent = SACAgent(state_dim=state_dim, action_dim=action_dim, memo_capacity=memory_size, alpha=3e-4, beta=3e-4,
                 gamma=0.99, tau=0.005, layer1_dim=64, layer2_dim=64, batch_size=256) #TODO

best_reward = 0
Num_episodes = 10000
Num_step = 200
Reward_buffer = []
Plot_reward = True

# the directory to save model
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
timestamp = time.strftime('%Y%m%d-%H%M%S')

for episode_i in range(Num_episodes):
    state, others = env.reset()
    episode_reward = 0
    for step_i in range(Num_step):
        action = agent.get_action(state) # TODO
        state_, reward, done, trunc, info = env.step(action)
        agent.add_memory(state, action, reward, state_, done) # TODO
        episode_reward += reward
        state = state_
        agent.update() #TODO

        if done:
            break
    Reward_buffer.append(episode_reward)
    avg_reward = np.mean(Reward_buffer)

    # Save model
    if avg_reward > best_reward:
        best_reward = avg_reward
        torch.save(agent.actor.state_dict(), model + f'sac_actor_{timestamp}.pth')
        print(f"...saving model with best reward:{best_reward}")

    print(f'Episode: {episode_i}, Reward: {episode_reward}')

env.close()

if Plot_reward:
    plt.plot(np.arange(len(Reward_buffer)), Reward_buffer, color='purple', alpha=0.5, label='Reward')
    plt.plot(np.arange(len(Reward_buffer)), gaussian_filter(Reward_buffer, sigma=5), color='red', linewidth=2)
    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig(f"Reward -{scenario}-{timestamp}.png", format='png')
    plt.show()