import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

from utils import episode_reward_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

def compute_returns(rewards, next_value, discount):
    """ Compute returns based on episode rewards.

    Parameters
    ----------
    rewards : list of float
        Episode rewards.
    next_value : float
        Value of state the episode ended in. Should be 0.0 for terminal state, bootstrapped value otherwise.
    discount : float
        Discount factor.

    Returns
    -------
    list of float
        Episode returns.
    """

    # TODO (3.)

    T = len(rewards)
    returns = np.zeros(T,dtype=np.float32)
    for t in range(T):
        G_t = 0
        for k in range(t, T):
            G_t += discount**(k-t) * rewards[k]
        G_t += (discount**(T-t)) * next_value
        returns[t] = G_t
    # print(f"compute_returns called - Returns length: {len(returns)}")
    return returns.tolist() # list


class TransitionMemory:
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE) at the end of an episode."""

    def __init__(self, gamma):

        # TODO (2.)
        self.gamma = gamma
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.returns_lst = []
        self.trajectory_start = 0
        # self.trajectory_end = 0

    def put(self, obs, action, reward, logprob):
        """Put a transition into the memory."""

        # TODO
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(logprob)

    def get(self):
        """Get all stored transition attributes in the form of lists."""

        # TODO
        return self.observations, self.actions, self.rewards, self.log_probs

    def clear(self):
        """Reset the transition memory."""

        # TODO
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.returns_lst = []  # 确保清空
        self.trajectory_start = 0

    def finish_trajectory(self, next_value):
        """Call on end of an episode. Will perform episode return or advantage or generalized advantage estimation (later exercise).
        
        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state.
        """

        # TODO
        observations, actions, rewards, log_probs = self.get()

        print(f"logprobs called - Returns length: {len(self.log_probs)}")

        returns = compute_returns(rewards[self.trajectory_start:], next_value, self.gamma)
        self.trajectory_start = len(self.rewards)
        self.returns_lst.extend(returns)  # 累积所有 episode 的 returns

        print(f"return_lst called - Returns length: {len(self.returns_lst)}")




class ActorNetwork(nn.Module):
    """Neural Network used to learn the policy."""

    def __init__(self, num_observations, num_actions):
        super(ActorNetwork, self).__init__()

        # TODO (1.)
        self.fc1 = nn.Linear(num_observations, 128)
        self.fc2 = nn.Linear(128, num_actions)
        self.relu = nn.ReLU()
        self.Softmax = nn.Softmax(dim=-1) # better for single sample and batch

    def forward(self, obs):

        # TODO
        x =self.relu(self.fc1(obs))
        x = self.fc2(x) # x is logits
        action_probs = self.Softmax(x) # convert to probability
        return action_probs


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

        # TODO (6.)
        obs, _ = self.env.reset()

        # For plotting
        overall_rewards = []
        episode_rewards = []

        episodes_counter = 0

        for timestep in range(1, total_timesteps + 1):

            # TODO Do one step, put into transition buffer, and store reward in episode_rewards for plotting
            action, logprob = self.predict(obs, train_returns=True)
            obs_, reward, terminated, truncated, info = self.env.step(action)
            episode_rewards.append(reward)
            self.memory.put(obs, action, reward, logprob)
            obs = obs_

            if terminated or truncated:

                # TODO reset environment
                obs, _ = self.env.reset()
                # TODO finish trajectory
                self.memory.finish_trajectory(0.0)
                overall_rewards.append(sum(episode_rewards))
                print(f"Episode ended at timestep {timestep}, total reward: {sum(episode_rewards)}, overall_rewards length: {len(overall_rewards)}")
                episode_rewards = []
                episodes_counter += 1

                if episodes_counter == self.episodes_update:
                    # TODO optimize the actor

                    loss = self.calc_actor_loss(self.memory.log_probs, self.memory.returns_lst)
                    self.optim_actor.zero_grad()


                    loss.backward()
                    self.optim_actor.step()

                    # Clear memory
                    episodes_counter = 0
                    self.memory.clear()

            # Episode reward plot
            if timestep % 500 == 0:
                print(f"Timestep {timestep}, overall_rewards length: {len(overall_rewards)}, rewards: {overall_rewards[-5:] if overall_rewards else []}")
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1,
                                    wait=timestep == total_timesteps)

    @staticmethod
    def calc_actor_loss(logprob_lst, return_lst):
        """Calculate actor "loss" for one batch of transitions."""

        # TODO (5.)

        if not all(isinstance(x, torch.Tensor) for x in logprob_lst):
            raise ValueError(f"logprob_lst contains non-tensor elements: {[type(x) for x in logprob_lst]}")

            # Stack list of tensors into a single tensor
        logprob_lst = torch.stack(logprob_lst).to(device)
        return_lst = torch.FloatTensor(return_lst).to(device)

        # Debug shapes and types
        print(
            f"logprob_lst shape: {logprob_lst.shape}, dtype: {logprob_lst.dtype}, requires_grad: {logprob_lst.requires_grad}")
        print(
            f"return_lst shape: {return_lst.shape}, dtype: {return_lst.dtype}, requires_grad: {return_lst.requires_grad}")

        # Ensure shapes match
        if logprob_lst.shape != return_lst.shape:
            raise ValueError(f"Shape mismatch: logprob_lst {logprob_lst.shape}, return_lst {return_lst.shape}")

        actor_loss = -torch.mean(logprob_lst * return_lst)
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

        # TODO (4.)
        obs_tensor = torch.FloatTensor(obs).to(device)
        action_prob = self.actor_net(obs_tensor)
        policy_dist = Categorical(action_prob)
        action = policy_dist.sample()

        if train_returns:

            # TODO Return action, logprob
            log_prob = policy_dist.log_prob(action) # Type -> Tensor
            return action.detach().cpu().item(), log_prob
        else:

            # TODO Return action
            return action.detach().cpu().item(), None


if __name__ == '__main__':
    env_id = "CartPole-v1"
    _env = gym.make(env_id)
    vpg = VPG(_env)
    vpg.learn(3000)
