import torch
import torch.nn as nn


class BasePriorModel(nn.Module):
    def __init__(self, log_prior_fn: callable):
        super().__init__()
        self.log_prior_fn = log_prior_fn

    def log_prior_prob(self, ):
        return torch.sum(torch.cat([self.log_prior_fn(p).sum()[None] for p in self.parameters()]))

    def get_negative_log_prob(self, x):
        return self.forward(x)


class MLPEnergyModel(BasePriorModel):
    """
    MLP Energy based model
    """

    def __init__(self, in_features, log_prior_fn):
        super().__init__(log_prior_fn)
        self.energy_net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 2, out_features=in_features // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 2, out_features=1),
        )
        self.sample_theta = True

    def forward(self, x):
        return self.energy_net(x)


class GaussianPriorModel(BasePriorModel):
    """
    Gaussian Prior Model
    """

    def __init__(self, in_features, log_prior_fn, **kwargs):
        super().__init__(log_prior_fn)
        self.aux = nn.Parameter(torch.randn(1))
        if kwargs["fix_mu"]:
            self.register_buffer('mu', torch.zeros(in_features), persistent=False)
        else:
            self.mu = nn.Parameter(torch.randn(in_features))

        # self.raw_sigma = nn.Parameter(torch.randn(in_features))
        if kwargs["fix_scale"]:
            self.register_buffer('raw_sigma', kwargs["scale_init"] * torch.ones(in_features),
                                 persistent=False)  # CHECK IT!
        else:
            self.raw_sigma = nn.Parameter(kwargs["scale_init"] * torch.ones(in_features), requires_grad=False)

        self.sample_theta = False

    def forward(self, x):
        return -torch.distributions.Normal(loc=self.mu, scale=torch.nn.functional.softplus(self.raw_sigma)).log_prob(
            x).sum(-1) + 0. * self.aux

    def sample(self, x):
        return torch.distributions.Normal(loc=self.mu, scale=torch.nn.functional.softplus(self.raw_sigma)).sample()


class MixtureofGaussianPriorModel(BasePriorModel):
    """
    Mixture of Gaussian Prior Model
    """

    def __init__(self, in_features, log_prior_fn, n_modes, **kwargs):
        super().__init__(log_prior_fn)

        self.mu = nn.Parameter(torch.randn(n_modes, in_features))
        self.raw_sigma = nn.Parameter(torch.randn(n_modes, in_features))
        self.logits = nn.Parameter(torch.randn(n_modes, ))
        self.mix = torch.distributions.Categorical(logits=self.logits)
        self.distributions = torch.distributions.Independent(
            torch.distributions.Normal(loc=self.mu, scale=torch.nn.functional.softplus(self.raw_sigma)), 1
        )
        self.distribution = torch.distributions.MixtureSameFamily(mixture_distribution=self.mix,
                                                                  component_distribution=self.distributions)

        self.sample_theta = False

    def forward(self, x):
        self.distribution._component_distribution.base_dist.scale = torch.nn.functional.softplus(self.raw_sigma)
        self.mix.logits = self.logits
        return -self.distribution.log_prob(x)


class MLPSimpleEnergyModel(BasePriorModel):
    """
    MLP Energy based model
    """

    def __init__(self, in_features, log_prior_fn):
        super().__init__(log_prior_fn)
        self.sample_theta = True
        self.energy_net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=30),
            nn.LeakyReLU(),
            nn.Linear(in_features=30, out_features=1),
        )

    def forward(self, x):
        return self.energy_net(x)
