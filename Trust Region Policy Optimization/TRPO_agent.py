import copy
from copy import deepcopy

import numpy as np
from torch import nn
import torch
from torch.distributions import Normal


device = torch.device("cude" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)  # mean
        self.fc_std = nn.Linear(hidden_dim, action_dim)  # variance
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Pendulum [-2, 2]
        self.softplus = nn.Softplus()

    def forward(self, state):
        """
        Pendulum-v1 model is continuous action space, Actor network outputs high dimensional Gaussian distribution.
        factorized Gaussian policy : action dimension = n,we assume independent Gaussian distributions for each dimension.
                                    This means the joint distribution over actions is a product of n independent Gaussians
                                     — one for each action dimension.
        :param state:
        :return:
        """
        x = self.relu(self.fc1(state))  #
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * 2  # 缩放到 Pendulum 的动作范围 [-2, 2]
        std = self.softplus(self.fc_std(x)) + 1e-3  # softplus 激活函数+ epsilon 确保标准差 > 0
        # for Pendulum-v1 action space has 1 dimension, so output is mean and variance of one Gaussian Distribution

        return mean, std

    def select_action(self, state):
        with torch.no_grad():
            mu, sigma = self.forward(state)
            normal_dist = Normal(mu, sigma)  # assume Gaussian Distribution
            # 横轴(x)：是随机变量的取值，在强化学习中就是你可能选择的动作（action），比如在 Pendulum 中表示施加的转矩
            # 纵轴 (y)：是 对应横轴值的概率密度（PDF 值），也就是这个动作“出现”的可能性大小（不是概率，而是密度）。
            action = normal_dist.sample()
            action = action.clamp(-2,2) # action represents the torque , action ->[-2,2] continuous action space
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)

        return value

class ReplayMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []

    def push(self, state, action, reward, value, done):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap.append(value)
        self.done_cap.append(done)

    def sample(self):
        num_states = len(self.state_cap)
        batch_start_index = np.arange(0, num_states, self.batch_size) # batch_size = 3 [0,2,4]
        memory_indices = np.arange(num_states, dtype=np.int32) # [0, 1, 2,3,4,5,6]
        np.random.shuffle(memory_indices) # 将索引[0, 1, 2,3,4,5,6] 打乱[2,5,3,1,4,6]
        #batches = [
        #    memory_indices[0:2],  # [2, 5]
        #    memory_indices[2:4],  # [3, 1]
        #    memory_indices[4:6]  # [4, 6]
        #]
        # batches = [[2, 5], [3, 1], [4, 6]]
        batches = [memory_indices[i:i+self.batch_size] for i in batch_start_index] # batch = [memory_indices[0:2]]

        return np.array(self.state_cap), np.array(self.action_cap), np.array(self.reward_cap), np.array(self.value_cap), np.array(self.done_cap), batches

    def clear(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []


class trpo_Agent:
    def __init__(self, state_dim, action_dim, batch_size):
        self.Lr_actor = 3e-4
        self.Lr_critic = 3e-4
        self.gamma = 0.99
        self.lamda = 0.95
        self.Num_epoch = 10
        self.epsilon_clip = 0.2

        self.kl_constraint = 0.005
        self.alpha = 0.5

        self.batch_size = batch_size

        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.Lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.Lr_critic)
        self.replay_memory = ReplayMemory(batch_size)

    def get_action(self,state):
        # add dim from state = [cos(θ), sin(θ), θ̇]to state = [[cos(θ), sin(θ), θ̇]]
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor.select_action(state)
        value = self.critic.forward(state)

        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]

    def compute_returns(self, rewards, values, dones):
        advantage = 0
        num_transitions = len(rewards)
        memo_advantage = np.zeros(num_transitions, dtype=np.float32)
        returns = np.zeros_like(rewards)
        for step in reversed(range(num_transitions)):
            if step == num_transitions - 1:
                next_value = values[-1]
            else:
                next_value = values[step + 1]
            next_is_not_terminal = 1 - int(dones[step])
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_value - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lamda * advantage
            returns[step] = advantage + values[step]
            memo_advantage[step] = advantage
        memo_advantage = (memo_advantage - memo_advantage.mean()) / (memo_advantage.std() + 1e-8)

        return memo_advantage

    def update(self):
        self.old_actor.load_state_dict(self.actor.state_dict()) # make new actor to old then

        for epoch_i in range(self.Num_epoch):
            # action = [a1,a2,a3,a4...an], value_cap = [v1,v2,v3,..vn]
            memo_states, memo_actions , memo_rewards, memo_values, memo_done, batches = self.replay_memory.sample()
            memo_advantage = self.compute_returns(memo_rewards, memo_values, memo_done)


            with torch.no_grad():
                memo_advantage_tensor = torch.tensor(memo_advantage).unsqueeze(1).to(device) # add dim
                memo_values_tensor = torch.tensor(memo_values).to(device)

            memo_states_tensor = torch.FloatTensor(memo_states).to(device)
            memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)


            for batch in batches:
                with torch.no_grad():
                    old_mu, old_sigma = self.old_actor(memo_states_tensor[batch]) # batches = [[2, 5], [3, 1], [4, 6]]
                    old_pi = Normal(old_mu, old_sigma)
                # 这里是得到每个动作的高斯分布，然后再根据动作(高斯分布的横轴)到概率分布(纵轴)
                batch_old_probs_tensor = old_pi.log_prob(memo_actions_tensor[batch]) # batch_old_probs_tensor = log[PI_theta]

                mu, sigma = self.actor(memo_states_tensor[batch])
                pi = Normal(mu, sigma)
                # batch_probs_tensor = pi.log_prob(memo_actions_tensor[batch]) # batch_probs_tensor = log[PI_old]

                #in the first epoch_i namely epoch_i=0, batch_probs_tensor = batch_old_probs_tensor because two networks have same parameters
                #in paper "PI_theta/PI_old" --> trick: exp(PI_theta)/exp(PI_old) = torch.exp(log[PI_theta] - log[PI_old])
                #ration = torch.exp(batch_probs_tensor - batch_old_probs_tensor)
                #surr1 = ration * memo_advantage_tensor[batch]
                #surr2 = torch.clamp(ration, 1- self.epsilon_clip, 1 + self.epsilon_clip) * memo_advantage_tensor[batch]

                #actor_loss = -torch.min(surr1,surr2).mean()

                # R_t = A_t + V(s_t)
                batch_returns = memo_advantage_tensor[batch] + memo_values_tensor[batch]
                batch_old_values = self.critic(memo_states_tensor[batch])
                # We use a value function loss of the form (V_theta(s_t) - R_t)^2
                critic_loss = nn.MSELoss()(batch_old_values, batch_returns)

                #self.actor_optimizer.zero_grad()
                #actor_loss.backward()
                #self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                """
                policy_learn
                
                states = memo_states_tensor
                actions = memo_actions_tensor
                old_action_dists = old_pi
                old_log_probs = batch_old_probs_tensor
                advantages = memo_advantage_tensor
                """
                # L
                surrogate_obj = self.compute_surrogate_obj(memo_states_tensor[batch],memo_actions_tensor[batch],memo_advantage_tensor[batch],batch_old_probs_tensor,self.actor)
                # delta(L)
                grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
                obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
                # conjugate_graddient method for x = H^(-1)g
                descent_direction = self.conjugate_gradient(obj_grad, memo_states_tensor[batch], old_pi)
                Hd = self.hessian_matrix_vector_product(memo_states_tensor[batch], old_pi, descent_direction)

                max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction,Hd) + 1e-8))
                new_para = self.line_search(memo_states_tensor[batch], memo_actions_tensor[batch], memo_advantage_tensor[batch],batch_old_probs_tensor,old_pi, descent_direction * max_coef)
                torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

        self.replay_memory.clear()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        mu, sigma = self.actor(states)
        new_action_dists = Normal(mu, sigma)
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])

        return grad2_vector

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        mu, sigma = actor(states)
        pi = Normal(mu, sigma)
        log_probs = pi.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)

        return torch.mean(ratio * advantage)

    def conjugate_gradient(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r,r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product (states, old_action_dists, p)
            alpha = rdotr / (torch.dot(p, Hp) + 1e-8)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r,r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        best_para = old_para
        best_obj = old_obj

        for i in range(15):
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            mu, sigma = new_actor(states)
            new_action_dists = Normal(mu, sigma)

            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)

            if new_obj > old_obj and kl_div < self.kl_constraint:
                best_para = new_para
                best_obj = new_obj
        # torch.nn.utils.convert_parameters.vector_to_parameters(best_para, self.actor.parameters())
        return best_para



    def save_policy(self):
        torch.save(self.actor.state_dict(), "ppo_policy_pendulum_v1.para")











