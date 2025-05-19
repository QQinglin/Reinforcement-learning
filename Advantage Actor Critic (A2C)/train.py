import argparse
import gymnasium as gym

from utils import visualize_agent
from vpg import VPG
from a2c import A2C


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', '-agent', type=str, default='a2c', choices=['vpg', 'a2c'])
    parser.add_argument('--steps', '-steps', type=int, default=5000)
    parser.add_argument('--lr_actor', '-lr_actor', type=float, default=0.01)
    parser.add_argument('--lr_critic', '-lr_critic', type=float, default=0.001)
    parser.add_argument('--gamma', '-gamma', type=float, default=0.99)
    parser.add_argument('--batch', '-batch', type=int, default=500)
    parser.add_argument('--use_gae', '-use_gae', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _args = parse()

    _env = gym.make("CartPole-v1")

    if 'vpg' == _args.agent:
        _agent = VPG(_env, gamma=_args.gamma, lr=_args.lr_actor)
    else:
        _agent = A2C(_env, gamma=_args.gamma, lr_actor=_args.lr_actor, lr_critic=_args.lr_critic,
                     batch_size=_args.batch, use_gae=_args.use_gae)

    _agent.learn(_args.steps)

    # Visualize the agent
    visualize_agent(gym.make("CartPole-v1", render_mode='human'), _agent)
