import argparse
import gymnasium as gym
from ppo import PPO
from utils import visualize_agent



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', '-agent', type=str, default='ppo', choices=['ppo', 'a2c'])
    parser.add_argument('--episodes', '-episodes', type=int, default=10000)
    parser.add_argument('--Num_steps_per_episode', '-Num_steps_per_episode', type=int, default=200)
    parser.add_argument('--lr_actor', '-lr_actor', type=float, default=0.003)
    parser.add_argument('--lr_critic', '-lr_critic', type=float, default=0.003)
    parser.add_argument('--gamma', '-gamma', type=float, default=0.99)
    parser.add_argument('--batch', '-batch', type=int, default=64)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _args = parse()

    _env = gym.make('Pendulum-v1')

    if 'ppo' == _args.agent:
        _agent = PPO(_env)
    #else:
        #_agent = A2C(_env, gamma=_args.gamma, lr_actor=_args.lr_actor, lr_critic=_args.lr_critic,
                     #batch_size=_args.batch, use_gae=_args.use_gae)

    _agent.learn(_args.episodes, _args.Num_steps_per_episode)

    # Visualize the agent
    visualize_agent(gym.make('Pendulum-v1', render_mode='human'), _agent)
