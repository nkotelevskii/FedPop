import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNCifar100_shared(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, kwargs["shared_embedding_size"])  # 128

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.drop((F.relu(self.fc2(x))))
        return x


class MLP_MNIST_shared(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_input = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, kwargs["shared_embedding_size"])

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
        x = self.relu(x)
        return x


class BigBaseConvNetMNIST_shared(nn.Module):
    def __init__(self, shared_embedding_size):
        super(BigBaseConvNetMNIST_shared, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(1024, shared_embedding_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return x


# class BigBaseConvNetCIFAR_shared(nn.Module):
#     def __init__(self, shared_embedding_size):
#         super(BigBaseConvNetCIFAR_shared, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(64, 64, 5)
#         self.fc1 = nn.Linear(1600, shared_embedding_size)  # 384
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         return x


class BigBaseConvNetCIFAR_shared(nn.Module):
    def __init__(self, shared_embedding_size):
        super(BigBaseConvNetCIFAR_shared, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, shared_embedding_size)  # 64

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class CNN_shared(nn.Module):
    """
    CNN from McMahan et al., 2016
    """

    def __init__(self, shared_embedding_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        return y


class LeNet5_shared(nn.Module):
    """
    LeNet5 for MNIST
    """

    def __init__(self, shared_embedding_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, shared_embedding_size)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


class TestModel_shared(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3)
        self.linear1 = nn.Linear(12, 12)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(12, 13)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(13, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        h = self.conv(x)
        h = self.flatten(h)

        h = self.linear1(h)
        h = self.relu1(h)

        h = self.linear2(h)
        h = self.relu2(h)

        h = self.linear3(h)

        return h


class MLPClassificationModel_shared(nn.Module):
    def __init__(self, shared_embedding_size):
        super().__init__()
        self.linear1 = nn.Linear(2, 5)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(5, 5)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(5, shared_embedding_size)

    def forward(self, x):
        h = self.linear1(x)
        h = self.relu1(h)

        h = self.linear2(h)
        h = self.relu2(h)

        h = self.linear3(h)

        return h


class MLPRegressionModel_shared(nn.Module):
    def __init__(self, shared_embedding_size):
        super().__init__()
        activation = nn.ReLU
        self.linear1 = nn.Linear(1, 5)
        self.act1 = activation()
        self.linear2 = nn.Linear(5, 5)
        self.act2 = activation()
        self.linear3 = nn.Linear(5, 5)
        self.act3 = activation()
        self.linear4 = nn.Linear(5, shared_embedding_size)

    def forward(self, x):
        h = self.linear1(x)
        h = self.act1(h)

        h = self.linear2(h)
        h = self.act2(h)

        h = self.linear3(h)
        h = self.act3(h)

        h = self.linear4(h)
        return h


class MultivariateCovariance_shared(nn.Module):
    def __init__(self, shared_embedding_size):
        super().__init__()
        self.lower_diag = nn.Parameter(torch.randn(3, dtype=torch.float32))
        self.register_buffer('dummy_param', torch.empty(0))

    def forward(self, x):
        z = torch.zeros(size=[1], device=self.dummy_param.device)
        diag = 1. + nn.functional.elu(self.lower_diag[:2])
        tril = self.lower_diag[2:]
        scale_tril = torch.stack([
            diag[0:1], z,
            tril[0:1], diag[1:2]
        ], dim=-1).view(2, 2)
        return scale_tril.view(-1)
