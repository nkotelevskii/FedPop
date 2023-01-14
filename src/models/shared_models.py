import torch

from .raw_models.shared import LeNet5_shared, TestModel_shared, MLPClassificationModel_shared, \
    MLPRegressionModel_shared, MultivariateCovariance_shared, CNN_shared, BigBaseConvNetMNIST_shared, \
    BigBaseConvNetCIFAR_shared, MLP_MNIST_shared, CNNCifar100_shared

__all__ = ['LeNet5', 'TestModel', 'MLPClassificationModel', 'MLPRegressionModel', 'MultivariateCovariance', 'CNN',
           'BigBaseConvNetMNIST', 'BigBaseConvNetCIFAR', 'MLP_MNIST', 'CNNCifar100']


class BaseCentralizedModelMixin:
    """
    This class provides an additional functional to be used in mixin models.
    """

    def set_log_prior_fn(self, log_prior_fn):
        self.log_prior_fn = log_prior_fn

    def log_prior_prob(self, ):
        return torch.sum(torch.cat([self.log_prior_fn(p).sum()[None] for p in self.parameters()]))


################################################################################
#############################  Mixins are below  ###############################
################################################################################

class CNNCifar100(CNNCifar100_shared, BaseCentralizedModelMixin):
    pass


class MLP_MNIST(MLP_MNIST_shared, BaseCentralizedModelMixin):
    pass


class BigBaseConvNetMNIST(BigBaseConvNetMNIST_shared, BaseCentralizedModelMixin):
    pass


class BigBaseConvNetCIFAR(BigBaseConvNetCIFAR_shared, BaseCentralizedModelMixin):
    pass


class CNN(CNN_shared, BaseCentralizedModelMixin):
    pass


class LeNet5(LeNet5_shared, BaseCentralizedModelMixin):
    pass


class TestModel(TestModel_shared, BaseCentralizedModelMixin):
    pass


class MLPClassificationModel(MLPClassificationModel_shared, BaseCentralizedModelMixin):
    pass


class MLPRegressionModel(MLPRegressionModel_shared, BaseCentralizedModelMixin):
    pass


class MultivariateCovariance(MultivariateCovariance_shared, BaseCentralizedModelMixin):
    pass
