import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from ..utils import count_parameters


class PersonalModel(nn.Module):
    def __init__(self, model: nn.Module, backbone_model: nn.Module):
        """

        Parameters
        ----------
        model: nn.Module
            Local (personalized) model
        backbone_model: nn.Module
            Model, which will be used for preprocessing of the input. In particular, could be identical
        """
        super().__init__()
        self.model = model
        self.backbone_model = backbone_model
        self.backbone_n_parameters = count_parameters(model=self.backbone_model)
        self.theta = {
            "with_ll": parameters_to_vector(self.model.parameters()).detach(),
            "without_ll": parameters_to_vector(self.model.parameters()).detach()
        }

    def forward(self, x):
        return self.model(x)
