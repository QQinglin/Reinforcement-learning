from collections import deque

import torch
import numpy as np
from torch.onnx.symbolic_opset9 import tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
######################
## For A2c and PPO  ##
######################

def compute_advantages(returns, values):
    """ Compute episode advantages based on precomputed episode returns.

    Parameters
    ----------
    returns : list of float
        Episode returns calculated with compute_returns.
    values: list of float
        Critic outputs for the states visited during the episode

    Returns
    -------
    list of float
        Episode advantages.
    """

    # TODO 1.4: Compute the advantages using equation 1.
    # Convert to tensors but don't require gradients
    returns_tensor = torch.stack(returns)
    values_tensor = torch.stack(values)
    advantages = returns_tensor - values_tensor# [A_1, A_2, ..., A_T] # torch tensor
    return advantages


def compute_generalized_advantages(rewards, values, next_value, discount, lamb):
    """ Compute generalized advantages (GAE) of the episode.

    Parameters
    ----------
    rewards : list of float
        Episode rewards.
    values: list of float
        Episode state values.
    next_value : float
        Value of state the episode ended in. Should be 0.0 for terminal state, critic output otherwise.
    discount : float
        Discount factor.
    lamb: float
        Lambda parameter of GAE.

    Returns
    -------
    list of float
        Generalized advantages of the episode.
    """

    # TODO 1.8: Compute GAE using equation 3.
    T = len(rewards)
    generalized_advantages = [] # Initialize GAE list for each timestep [A_1, A_2, ..., A_T]

    deltas = np.zeros(T,dtype=np.float32) # TD error δ_t
    gae = 0
    for t in range(T):
        # V(s_{t+1}) is next_value if t is the last timestep, otherwise use values[t+1]
        next_val = next_value if t == T - 1 else values[t + 1]
        delta_t = rewards[t] + discount * next_val - values[t]
        deltas[t] = delta_t

    for t in range(T - 1, -1, -1):  # Iterate from T-1 down to 0
        # GAE formula: A_t = δ_t + (γ * λ) * A_{t+1}
        gae = deltas[t] + (discount * lamb) * gae
        generalized_advantages.extend(gae) # generalized_advantages is a list containing tensor
    return generalized_advantages



class TransitionMemoryAdvantage:
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE)
     at the end of an episode."""

    def __init__(self, gamma, lamb, use_gae):
        self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst = [], [], [], [], []
        self.gamma = gamma
        self.traj_start = 0

        self.lamb = lamb
        self.use_gae = use_gae
        # TODO 1.2: Define additional datastructures
        # define additional datastructures
        self.value_lst = []
        self.advantage_lst = []

    def put(self, obs, action, reward, logprob, value):
        """Put a transition into the memory."""
        self.obs_lst.append(obs)
        self.action_lst.append(action)
        self.reward_lst.append(reward)
        self.logprob_lst.append(logprob)  # torch tensor
        self.value_lst.append(value)  # keep as torch tensor with grad

        # TODO 1.2: append to new datastructures



    def get(self):
        """Get all stored transition attributes in the form of lists."""
        # TODO 1.2: Return new datastructures

        return self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst, self.value_lst, self.advantage_lst

    def clear(self):
        """Reset the transition memory."""
        self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst = [], [], [], [], []
        self.traj_start = 0

        # TODO 1.2: Clear new datastructures
        self.advantage_lst = []
        self.value_lst = []

    def finish_trajectory(self, next_value=0.0):
        """Call on end of an episode. Will perform episode return and advantage or generalized advantage estimation.

        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state, critic output otherwise.
        """
        reward_traj = self.reward_lst[self.traj_start:] # [r1, ..., r_T]

        rewards_tensor = torch.FloatTensor(reward_traj).to(device)  # Convert to 1D tensor
        return_traj = compute_returns(rewards_tensor, next_value, self.gamma) # discount reward γr [G_1,... G_(T-1),G_T]
        print(f"return_traj : {type(return_traj)}")
        self.return_lst.extend(return_traj) # # G_1,... G_(T-1),G_T  return_lst is a list containing tensor

        # TODO 1.2: Extract values before updating trajectory termination counter
        # Safely convert value objects to numeric values

        value_traj = []
        value_traj.extend(self.value_lst[self.traj_start:] )

        # # for debug
        # for i, elem in enumerate(value_traj):
        #     if not isinstance(elem, torch.Tensor):
        #         print(f"Warning: Element at index {i} is not a torch.Tensor, it is {type(elem)} with value {elem}")
        #     else:
        #         print(
        #             f"Element at index {i} is a torch.Tensor, shape: {elem.shape}, dtype: {elem.dtype}, requires_grad: {elem.requires_grad}")

        #for v in self.value_lst[self.traj_start:]:
            #value_cpu = v.detach().cpu().item()  # Detach, move to CPU, and convert to float
            #value_traj.append(value_cpu)  # Append CPU float value to value_traj

        self.traj_start = len(self.reward_lst)


        if self.use_gae:
            traj_adv = compute_generalized_advantages(rewards_tensor, value_traj, next_value, self.gamma, self.lamb)
        else:
            traj_adv = compute_advantages(return_traj, value_traj)

        # Prevent trying to backward through graph a second time
        traj_adv = [adv.detach() for adv in traj_adv]

        # TODO 1.2: Append computed advantage to new datastructure
        self.advantage_lst.extend(traj_adv)  # gae list [A_1,... A_(T-1),A_T]



##############
## For VPG  ##
##############

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

    return_lst = []
    ret = next_value # scalar
    for reward in reversed(rewards): # rewards -> tensor
        ret = reward + discount * ret
        return_lst.append(ret)
    return return_lst[::-1]  # return_lst is a list containing tensor


class TransitionMemory:
    """Datastructure to store episode transitions and perform return at the end of an episode."""

    def __init__(self, gamma):
        self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst = [], [], [], [], []
        self.gamma = gamma
        self.traj_start = 0

    def put(self, obs, action, reward, logprob):
        """Put a transition into the memory."""
        self.obs_lst.append(obs)
        self.action_lst.append(action)
        self.reward_lst.append(reward)
        self.logprob_lst.append(logprob)

    def get(self):
        """Get all stored transition attributes in the form of lists."""
        return self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst

    def clear(self):
        """Reset the transition memory."""
        self.obs_lst, self.action_lst, self.reward_lst, self.logprob_lst, self.return_lst = [], [], [], [], []
        self.traj_start = 0

    def finish_trajectory(self, next_value=0.0):
        """Call on end of an episode. Will perform episode return or advantage or generalized advantage estimation (later exercise).

        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state.
        """
        reward_traj = self.reward_lst[self.traj_start:]
        return_traj = compute_returns(reward_traj, next_value, self.gamma)
        self.return_lst.extend(return_traj)
        self.traj_start = len(self.reward_lst)
