import argparse
import gymnasium as gym
from ddpg import DDPG
from utils import visualize_agent



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', '-agent', type=str, default='DDPG', choices=['DDPG', 'a2c'])
    parser.add_argument('--episodes', '-episodes', type=int, default=100)
    parser.add_argument('--Num_steps_per_episode', '-Num_steps_per_episode', type=int, default=200)
    parser.add_argument('--lr_actor', '-lr_actor', type=float, default=0.0003)
    parser.add_argument('--lr_critic', '-lr_critic', type=float, default=0.001)
    parser.add_argument('--gamma', '-gamma', type=float, default=0.99)
    parser.add_argument('--tau', '-tau', type=float, default=5e-3)
    parser.add_argument('--batch_size', '-batch_size', type=int, default=64)
    parser.add_argument('--memory_capacity', '-memory_capacity', type=int, default=10000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _args = parse()

    _env = gym.make('Pendulum-v1')

    if 'DDPG' == _args.agent:
        _agent = DDPG(_env, _args.gamma, _args.tau, _args.batch_size, _args.lr_actor, _args.lr_critic, _args.memory_capacity)
    #else:
        #_agent = A2C(_env, gamma=_args.gamma, lr_actor=_args.lr_actor, lr_critic=_args.lr_critic,
                     #batch_size=_args.batch, use_gae=_args.use_gae)

    _agent.learn(_args.episodes, _args.Num_steps_per_episode)

    # Visualize the agent
    visualize_agent(gym.make('Pendulum-v1', render_mode='human'), _agent)
