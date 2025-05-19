import numpy as np

from gridworld import GridWorldEnv
from util import value_function_policy_plot


class TDAgent:
    def __init__(self, env, discount_factor, learning_rate):
        self.env = env
        self.g = discount_factor
        self.lr = learning_rate

        self.num_actions = env.action_space.n

        # V[y, x] is value for grid position y, x, initialize to all zeros
        self.V = np.zeros(env.observation_space.high, dtype=np.float32) # create a [4, 4] zero array

        # uniform random policy[y, x, z], i.e. probability of action z when in grid position y, x is 1 / num_actions
        self.policy = np.ones((*env.observation_space.high, self.num_actions), dtype=np.float32) / self.num_actions

        # TODO 3: experiment with different (not fully random) policies
        for y in range(env.observation_space.high[0]):
            for x in range(env.observation_space.high[1]):
                self.policy[y, x, 0] = 0.1
                self.policy[y, x, 1] = 0.4
                self.policy[y, x, 2] = 0.1
                self.policy[y, x, 3] = 0.4

    def action(self, s):
        # TODO 2: Sample action following the policy
        s = tuple(s)
        action_prob = self.policy[s]

        return np.random.choice(self.num_actions, p=action_prob)  # random action

    def learn(self, n_timesteps=50000):
        s, _ = self.env.reset()
        episode_reward = 0
        # reward_buffer = np.empty(n_timesteps)
        # avg_reward_buffer = np.empty(n_timesteps)

        for i in range(n_timesteps):
            # TODO 1: Implement the agent-interaction loop
            # You will have to call self.update(...) at every step
            # Do not forget to reset the environment if you receive a 'terminated' signal
            action = self.action(s)
            next_state, reward, done, _, _= self.env.step(action)
            self.update(s,reward,next_state)
            s = next_state
            episode_reward += reward
            if done:
                break

            pass

    def update(self, s, r, s_):
        # TODO 1: Implement the TD estimation update rule
        self.V[s] = self.V[s] + self.lr * (r + self.g * self.V[s_] - self.V[s])

        return self.V[s]


if __name__ == "__main__":
    # Create Agent and environment
    td_agent = TDAgent(GridWorldEnv(), discount_factor=0.9, learning_rate=0.01)

    # Learn the state-value function for 100000 steps
    td_agent.learn(n_timesteps=100000)

    # Visualize V
    value_function_policy_plot(td_agent.V, td_agent.policy, td_agent.env.map)
