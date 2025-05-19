import torch.nn as nn

class NeuralNetwork(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, num_obs, num_actions):
        super(NeuralNetwork, self).__init__()

        # TODO 1.1: Implement the network structure
        self.layers = nn.Sequential(
            nn.Linear(in_features=num_obs, out_features=88),
            nn.Tanh(),
            nn.Linear(in_features=88, out_features=num_actions)
        )

    def forward(self, x): # tensor(batch_size, num_obs)
        # TODO 1.1: Implement the forward function, which returns the output net(x)
        return self.layers(x)