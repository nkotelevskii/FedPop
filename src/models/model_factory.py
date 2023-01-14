from .prior_models import MLPEnergyModel, MLPSimpleEnergyModel, GaussianPriorModel, MixtureofGaussianPriorModel
from .raw_models import MLPRegressionModel_personal, MLPClassificationModel_personal, IdenticalBackbone, \
    MultivariateMu_personal, IdenticalMLPBackbone, LeNet5Backbone, CNNBackbone, BigBaseConvNetBackbone, \
    MLPClassificationImages_personal, MLP_MNIST_personal
from .shared_models import MLPClassificationModel, MLPRegressionModel, MultivariateCovariance, LeNet5, CNN, \
    BigBaseConvNetMNIST, BigBaseConvNetCIFAR, CNNCifar100, MLP_MNIST

__all__ = ['set_model']

SHARED_MODELS_DICT = {
    "MLPClassificationModel": MLPClassificationModel,
    "MLPRegressionModel": MLPRegressionModel,
    "MultivariateCovariance": MultivariateCovariance,
    "LeNet5": LeNet5,
    "CNN": CNN,
    "BigBaseConvNetMNIST": BigBaseConvNetMNIST,
    "BigBaseConvNetCIFAR": BigBaseConvNetCIFAR,
    "CNNCifar100": CNNCifar100,
    "MLP_MNIST": MLP_MNIST,
}

PERSONAL_MODELS_DICT = {
    "MLPRegressionModel_personal": MLPRegressionModel_personal,
    "MLPClassificationModel_personal": MLPClassificationModel_personal,
    "MultivariateMu_personal": MultivariateMu_personal,
    "MLPClassificationImages_personal": MLPClassificationImages_personal,
    "MLP_MNIST_personal": MLP_MNIST_personal,
}

PRIOR_MODELS_DICT = {
    "MLPEnergyModel": MLPEnergyModel,
    "MLPSimpleEnergyModel": MLPSimpleEnergyModel,
    "GaussianPriorModel": GaussianPriorModel,
    "MixtureofGaussianPriorModel": MixtureofGaussianPriorModel
}

BACKBONE_MODELS_DICT = {
    "IdenticalBackbone": IdenticalBackbone,
    "IdenticalMLPBackbone": IdenticalMLPBackbone,
    "LeNet5Backbone": LeNet5Backbone,
    "CNNBackbone": CNNBackbone,
    "BigBaseConvNetBackbone": BigBaseConvNetBackbone,
}

MODELS_DICT = {
    'shared': SHARED_MODELS_DICT,
    'personal': PERSONAL_MODELS_DICT,
    'backbone': BACKBONE_MODELS_DICT,
    'prior': PRIOR_MODELS_DICT,
}


def set_model(model_name, model_params, model_type, device='cpu'):
    model = MODELS_DICT[model_type][model_name](**model_params)
    if device.find('cuda') > -1:
        model = model.to(device)

    return model
