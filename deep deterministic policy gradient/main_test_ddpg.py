import os

import pygame
import torch.nn as nn
import torch
import gymnasium as gym
import numpy as np

from main_ddpg import Num_episodes, Num_steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Initialize env
env = gym.make(id = 'Pendulum-v1',render_model = "rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

 # load para
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
actor_path = model + "ddpg_actor.pth"


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(state_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        x = torch.tanh(self.f3(x)) * 2  # [-2,2]
        return x

def process_frame(frame):
    frame = np.transpose(frame, (1,0,2))
    frame = pygame.surfarray.make_surface(frame)
    return pygame.transform.scale(frame,(width,height))


actor = Actor(state_dim, action_dim).to(device)
actor.load_state_dict(torch.load(actor_path))

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Test phase
Num_episodes = 30
Num_steps = 200
for episode_i in range(Num_episodes):
    state, others = env.reset()
    episode_reward = 0

    for tep_i in range(Num_steps):
        action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        state_, reward, done, truncation, info = env.step(action)
        state = state_

        frame = env.render()
        frame = process_frame(frame)
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock.tick(60) # FPS

    print(f"Episode {episode_i} Reward: {episode_reward}")

pygame.quit()
env.close()


