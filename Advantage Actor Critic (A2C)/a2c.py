import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from torch.distributed.argparse_util import env

from networks import ActorNetwork, CriticNetwork
from transition_memory import TransitionMemoryAdvantage
from utils import episode_reward_plot
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class A2C:
    """The Actor-Critic approach."""

    def __init__(self, env, batch_size=500, gamma=0.99, lamb=0.99, lr_actor=0.005, lr_critic=0.001, use_gae=True):
        """ Constructor.

        Parameters
        ----------
        env : gym.Environment
            The object of the gym environment the agent should be trained in.
        batch_size : int, optional
            Number of transitions to use for one optimization step.
        gamma : float, optional
            Discount factor.
        lamb : float, optional
            Lambda parameters of GAE.
        lr_actor : float, optional
            Learning rate used for actor Adam optimizer.
        lr_critic : float, optional
            Learning rate used for critic Adam optimizer.
        use_gae : bool, optional
            Use generalized advantage estimation
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.batch_size = batch_size
        self.env = env
        self.memory = TransitionMemoryAdvantage(gamma, lamb, use_gae)

        self.actor_net = ActorNetwork(self.obs_dim, self.act_dim)
        self.critic_net = CriticNetwork(self.obs_dim)
        self.optim_actor = optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.optim_critic = optim.Adam(self.critic_net.parameters(), lr=lr_critic)

    def learn(self, total_timesteps):
        """Train the actor-critic.

        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train the agent for.
        """
        obs, _ = self.env.reset()

        # For plotting
        overall_rewards = []
        episode_rewards = []

        episode_counter = 0
        for timestep in range(1, total_timesteps + 1):

            # TODO 1.7.a: Sample action and supplementary data, take step and save transition to buffer
            action, logprob, value = self.predict(obs,train_returns=True)  # action -> scalar
            obs_, reward, terminated, truncated, info = self.env.step(action)
            episode_rewards.append(reward)

            self.memory.put(obs, action, reward, logprob, value)

            if terminated or truncated:
                # TODO 1.7.b: Reset environment and call 'finish_trajectory' with correct 'next_value'
                action, logprob, next_value = self.predict(obs_, train_returns=False)
                self.memory.finish_trajectory(next_value)

                # Get data from memory
                _, _, _, logprob_lst, return_lst, value_lst, adv_lst = self.memory.get()

                # Train critic network
                critic_loss = self.calc_critic_loss(value_lst, return_lst)
                self.optim_critic.zero_grad()
                critic_loss.backward()
                self.optim_critic.step()

                # Train actor network
                actor_loss = self.calc_actor_loss(logprob_lst, adv_lst)
                self.optim_actor.zero_grad()
                actor_loss.backward()
                self.optim_actor.step()

                # Record rewards and reset
                overall_rewards.append(sum(episode_rewards))
                episode_rewards = []
                obs, _ = self.env.reset()
                self.memory.clear()
                episode_counter += 1
            else:
                # Update current obs
                obs = obs_

            if (timestep - episode_counter) == self.batch_size:
                # TODO 1.7.c: Call 'finish_trajectory' with correct 'next_value', calculate losses, perform updates
                self.memory.finish_trajectory(next_value)

                # Get transitions from memory
                _, _, _, logprob_lst, return_lst, value_lst, adv_lst = self.memory.get()

                # Train critic network
                critic_loss = self.calc_critic_loss(value_lst, return_lst)
                self.optim_critic.zero_grad()
                critic_loss.backward()
                self.optim_critic.step()

                # Train actor network
                actor_loss = self.calc_actor_loss(logprob_lst, adv_lst)
                self.optim_actor.zero_grad()
                actor_loss.backward()
                self.optim_actor.step()

                # Clear memory
                self.memory.clear()
                episode_counter = timestep

            # Episode reward plot
            if timestep % 500 == 0:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1)

    @staticmethod
    def calc_critic_loss(value_lst, return_lst):
        """Calculate critic loss for one batch of transitions."""

        # TODO 1.5: Compute the MSE between state values and returns

        # Debug shapes, types, and gradient information
        # print(f"value_lst (before conversion): {type(value_lst)}")
        # print(f"return_lst (before conversion): {type(return_lst)}")

        value_lst = torch.stack(value_lst).to(device)
        return_lst = torch.stack(return_lst).view(-1,1).to(device)

        # Debug shapes, types, and gradient information after conversion
        # print(
            # f"value_lst shape: {value_lst.shape}, dtype: {value_lst.dtype}, requires_grad: {value_lst.requires_grad}, grad_fn: {value_lst.grad_fn}")
        # print(
            # f"return_lst shape: {return_lst.shape}, dtype: {return_lst.dtype}, requires_grad: {return_lst.requires_grad}, grad_fn: {return_lst.grad_fn}")

        critic_loss = torch.mean(torch.square(return_lst - value_lst))
        return critic_loss

    @staticmethod
    def calc_actor_loss(logprob_lst, adv_lst):
        """Calculate actor "loss" for one batch of transitions."""

        # TODO 1.6: Adjust Compute actor loss (Hint: Very similar to VPG version)
        logprob_lst = torch.stack(logprob_lst).view(-1,1).to(device)

        # Convert advantages to tensor (no grad needed)
        advantages = torch.stack(adv_lst).to(device)

        # Debug shapes, types, and gradient information after conversion
        print(
        f"logprob_lst shape: {logprob_lst.shape}, dtype: {logprob_lst.dtype}, requires_grad: {logprob_lst.requires_grad}, grad_fn: {logprob_lst.grad_fn}")
        print(
        f"advantages shape: {advantages.shape}, dtype: {advantages.dtype}, requires_grad: {advantages.requires_grad}, grad_fn: {advantages.grad_fn}")

        #Compute actor loss
        actor_loss = -torch.mean(logprob_lst * advantages)
        return actor_loss


    def predict(self, obs, train_returns=False):
        """Sample the agents action based on a given observation.

        Parameters
        ----------
        obs : numpy.array
            Observation returned by gym environment
        train_returns : bool, optional
            Set to True to get log probability of decided action and predicted value of obs.
        """

        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs).to(device)
        action_prob = self.actor_net(obs_tensor)
        policy_dist = Categorical(action_prob)
        action = policy_dist.sample()

        log_prob = policy_dist.log_prob(action)

        # TODO 1.3 Evaluate the value function
        value = self.critic_net(obs_tensor)

        if train_returns:
            # Don't detach value - we need gradients for the critic
            return action.detach().cpu().item(), log_prob, value
        else:
            return action.detach().cpu().item(), log_prob, value