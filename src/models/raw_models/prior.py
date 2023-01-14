import torch
import torch.nn as nn


class TestEnergyModel(nn.Module):
    """
    The test architecture of Energy based model
    """

    def __init__(self, last_layers_size: int):
        """
        The model takes a vector of size num_params in the network, and outputs a scalar, proportional to log density
        Parameters
        ----------
        last_layers_size: int
            Number of parameters in last layers, which will be personalized for each of the users.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=last_layers_size, out_features=1)
        )

    def forward(self, x):
        return self.net(x)


class MultivariateGaussianEnergyModel(nn.Module):
    """
    Multivariate Gaussian Energy based model
    """

    def __init__(self, loc, covariance_matrix):
        super().__init__()
        self.energy_net = torch.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix,
        )

    def forward(self, x):
        return -self.energy_net.log_prob(x).sum()

    def sample(self, n):
        return self.energy_net.sample(n)
