import sys

import numpy as np
import torch

sys.path.insert(0, './')
from src.models.raw_models import MultivariateGaussianEnergyModel
from src.samplers import ULA
from src.utils.utils import seed_everything


def test_ula_multivariate_gaussian():
    seed_everything(42)
    loc = torch.tensor([10., 10., ], dtype=torch.float32)
    covariance_matrix = torch.tensor([[1., 0.95], [0.95, 2.]], dtype=torch.float32)
    gem = MultivariateGaussianEnergyModel(loc=loc, covariance_matrix=covariance_matrix)

    ula_sampler = ULA(step_size=0.02)
    initial_points = torch.randn((2000, 2), dtype=torch.float32, requires_grad=True)
    final_samples = ula_sampler.forward(
        current_state=initial_points,
        energy_model=gem,
        n_transitions=2000,
    ).cpu().detach().numpy()
    assert np.all(np.isclose(loc, final_samples.mean(0), rtol=1e-3, atol=1e-1)) and np.all(
        np.isclose(covariance_matrix, np.cov(m=final_samples.T), rtol=1e-3, atol=1e-1))
