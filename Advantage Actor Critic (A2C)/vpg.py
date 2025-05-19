import torch
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

from networks import ActorNetwork
from transition_memory import TransitionMemory
from utils import episode_reward_plot


class VPG:
    """The vanilla policy gradient (VPG) approach."""

    def __init__(self, env, episodes_update=5, gamma=0.99, lr=0.01):
        """ Constructor.

        Parameters
        ----------
        env : gym.Environment
            The object of the gym environment the agent should be trained in.
        episodes_update : int
            Number episodes to collect for every optimization step.
        gamma : float, optional
            Discount factor.
        lr : float, optional
            Learning rate used for actor and critic Adam optimizer.
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.memory = TransitionMemory(gamma)
        self.episodes_update = episodes_update

        self.actor_net = ActorNetwork(self.obs_dim, self.act_dim)
        self.optim_actor = optim.Adam(self.actor_net.parameters(), lr=lr)

    def learn(self, total_timesteps):
        """Train the VPG agent.

        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train the agent for.
        """

        obs, _ = self.env.reset()

        # For plotting
        overall_rewards = []
        episode_rewards = []

        episodes_counter = 0

        for timestep in range(1, total_timesteps + 1):

            action, logprob = self.predict(obs, train_returns=True)
            obs_, reward, terminated, truncated, _ = self.env.step(action)
            self.memory.put(obs, action, reward, logprob)
            episode_rewards.append(reward)

            # Update current obs
            obs = obs_

            if terminated or truncated:

                obs, _ = self.env.reset()

                overall_rewards.append(sum(episode_rewards))
                episode_rewards = []

                self.memory.finish_trajectory(0.0)

                episodes_counter += 1

                if episodes_counter == self.episodes_update:
                    # Get transitions from memory
                    _, _, _, logprob_lst, return_lst = self.memory.get()

                    # Calculate loss
                    loss = self.calc_actor_loss(logprob_lst, return_lst)

                    # Back-propagate and optimize
                    self.optim_actor.zero_grad()
                    loss.backward()
                    self.optim_actor.step()

                    # Clear memory
                    episodes_counter = 0
                    self.memory.clear()

            # Episode reward plot
            if timestep % 500 == 0:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1,
                                    wait=timestep == total_timesteps)

    @staticmethod
    def calc_actor_loss(logprob_lst, return_lst):
        """Calculate actor "loss" for one batch of transitions."""
        # Note: negative sign, as torch optimizers per default perform gradients descent, not ascent
        return -(torch.Tensor(return_lst) * torch.stack(logprob_lst)).mean()

    def predict(self, obs, train_returns=False):
        """Sample the agents action based on a given observation.

        Parameters
        ----------
        obs : numpy.array
            Observation returned by gym environment
        train_returns : bool, optional
            Set to True to get log probability of decided action and predicted value of obs.
        """
        probs = self.actor_net(torch.Tensor(obs))
        policy = Categorical(probs=probs)
        action = policy.sample()
        logprob = policy.log_prob(action)

        if train_returns:
            return action.item(), logprob
        else:
            return action.item()
