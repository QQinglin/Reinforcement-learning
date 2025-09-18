import numpy as np
from lib import GridworldMDP, print_value, print_deterministic_policy, init_value, random_policy

def policy_evaluation_one_step(mdp, V, policy, discount=0.99):
    """ Computes one step of policy evaluation.
    Arguments: MDP, value function, policy, discount factor
    Returns: Value function of policy
    """
    # Init value function array
    V_new = V.copy()

    # TODO: Write your implementation here
    delta = 0
    for s in range(mdp.num_states):
        Value = 0
        for a, action_probability in enumerate(policy[s]):
            for prob, next_state, reward, is_terminal in mdp.P[s][a]:
                Value += action_probability * prob * (reward + discount * V[next_state])
        delta = max(delta, np.abs(Value - V[s]))
        V_new[s] = Value
    return V_new

def policy_evaluation(mdp, policy, discount=0.99, theta=0.01):
    """ Computes full policy evaluation until convergence.
    Arguments: MDP, policy, discount factor, theta
    Returns: Value function of policy
    """
    # Init value function array
    V = init_value(mdp)

    # TODO: Write your implementation here
    while True:
        delta = 0
        for s in range(mdp.num_states):
            v = 0
            for a, action_probability in enumerate(policy[s]):
                for prob, next_state, reward, is_terminal in mdp.P[s][a]:
                    v+= action_probability * prob * (reward + discount * V[next_state])

            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_improvement(mdp, V, discount=0.99):
    """ Computes greedy policy w.r.t a given MDP and value function.
    Arguments: MDP, value function, discount factor
    Returns: policy
    """
    # Initialize a policy array in which to save the greed policy 
    policy = np.zeros((mdp.num_states, mdp.num_actions))

    # TODO: Write your implementation here
    def one_step_lookahead(state, V):
        A = np.zeros(mdp.num_actions)
        for a in range(mdp.num_actions):
            for prob, next_state, reward, is_terminal in mdp.P[state][a]:
                A[a] += prob * (reward + discount * V[next_state])
        return A

    while True:
        V = policy_evaluation_one_step(mdp, V, policy, discount)
        policy_stable = True
        for s in range(mdp.num_states):
            chosen_a = np.argmax(policy[s])
            action_value = one_step_lookahead(s, V)
            best_a = np.argmax(action_value)

            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(mdp.num_actions)[best_a]
        if policy_stable:
            return policy


def policy_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the policy iteration (PI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """

    # Start from random policy
    policy = random_policy(mdp)
    # This is only here for the skeleton to run.
    V = init_value(mdp)

    # TODO: Write your implementation here
    delta = 0.0
    new_V = policy_evaluation(mdp, policy, discount, theta)
    policy = policy_improvement(mdp, new_V, discount)
    V = new_V
    return V, policy

def value_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the value iteration (VI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """
    # Init value function array
    V = init_value(mdp)

    # TODO: Write your implementation here

    # Get the greedy policy w.r.t the calculated value function
    def one_step_lookahead(state,V):
        action = np.zeros(mdp.num_actions)
        for a in range(mdp.num_actions):
            for prob, next_state, reward, is_terminal in mdp.P[state][a]:
                action[a] += prob * (reward + discount * V[next_state])

        return action

    while True:
        delta = 0.0
        for s in range(mdp.num_states):
            A = one_step_lookahead(s,V)
            best_action = np.argmax(A)
            delta = max(delta, np.abs(best_action - V[s]))
            V[s] = best_action
        if delta < theta:
            break

    policy = np.zeros((mdp.num_states, mdp.num_actions))
    for s in range(mdp.num_states):
        A = one_step_lookahead(s,V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0

    return V, policy


if __name__ == "__main__":
    # Create the MDP
    mdp = GridworldMDP([6, 6])
    discount = 0.99
    theta = 0.01

    # Print the gridworld to the terminal
    print('---------')
    print('GridWorld')
    print('---------')
    mdp.render()

    # Create a random policy
    V = init_value(mdp)
    policy = random_policy(mdp)
    # Do one step of policy evaluation and print
    print('----------------------------------------------')
    print('One step of policy evaluation (random policy):')
    print('----------------------------------------------')
    V = policy_evaluation_one_step(mdp, V, policy, discount=discount)
    print_value(V, mdp)

    # Do a full (random) policy evaluation and print
    print('---------------------------------------')
    print('Full policy evaluation (random policy):')
    print('---------------------------------------')
    V = policy_evaluation(mdp, policy, discount=discount, theta=theta)
    print_value(V, mdp)

    # Do one step of policy improvement and print
    # "Policy improvement" basically means "Take greedy action w.r.t given a given value function"
    print('-------------------')
    print('Policy improvement:')
    print('-------------------')
    policy = policy_improvement(mdp, V, discount=discount)
    print_deterministic_policy(policy, mdp)

    # Do a full PI and print
    print('-----------------')
    print('Policy iteration:')
    print('-----------------')
    V, policy = policy_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)

    # Do a full VI and print
    print('---------------')
    print('Value iteration')
    print('---------------')
    V, policy = value_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)