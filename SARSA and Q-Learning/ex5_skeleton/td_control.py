import argparse

from utils import action_value_plot, test_agent
from gridworld import GridWorldEnv
from agents import SARSAAgent, QLearningAgent


def parse():
    parser = argparse.ArgumentParser()
    choices_environment = ['standard', 'cliffwalk']
    choices_agent = ['sarsa', 'qlearning']
    parser.add_argument('--environment', '-env', type=str, default='standard', choices=choices_environment)
    parser.add_argument('--agent', '-agent', type=str, default='sarsa', choices=choices_agent)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.05)
    parser.add_argument('--discount_factor', '-gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', '-eps', type=float, default=0.4)
    parser.add_argument('--n_timesteps', '-steps', type=int, default=200000)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    _args = parse()
    env = GridWorldEnv(map_name=_args.environment)

    if 'sarsa' == _args.agent:  # Train SARSA agent
        sarsa_agent = SARSAAgent(env, _args.discount_factor, _args.learning_rate, _args.epsilon)
        sarsa_agent.learn(n_timesteps=_args.n_timesteps)
        action_value_plot(sarsa_agent)
        print('Testing SARSA agent')
        test_agent(sarsa_agent, env, epsilon=0.1)
    elif 'qlearning' == _args.agent:  # Train Q-learning agent
        qlearning_agent = QLearningAgent(env, _args.discount_factor, _args.learning_rate, _args.epsilon)
        qlearning_agent.learn(n_timesteps=_args.n_timesteps)
        action_value_plot(qlearning_agent)
        print('Testing Q-Learning agent')
        test_agent(qlearning_agent, env, epsilon=0.1)
    else:
        raise NotImplementedError
