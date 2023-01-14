import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_MNIST_personal(nn.Module):
    def __init__(self, input_size, n_classes, ):
        super().__init__()
        self.layer_out = nn.Linear(input_size, n_classes)

    def forward(self, x):
        x = self.layer_out(x)
        return x


class MLPClassificationModel_personal(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size // 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size // 2, input_size // 2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 2, n_classes)

    def forward(self, x):
        h = self.linear1(x)
        h = self.relu1(h)

        h = self.linear2(h)
        h = self.relu2(h)

        h = self.linear3(h)

        return h


class MLPClassificationImages_personal(nn.Module):
    def __init__(self, input_size, n_classes=10):
        super().__init__()
        self.fc3 = nn.Linear(input_size, n_classes)

    def forward(self, x):
        x = self.fc3(x)
        return x


# class MLPClassificationImages_personal(nn.Module):
#     def __init__(self, input_size, n_classes=10):
#         super().__init__()
#         self.fc2 = nn.Linear(input_size, 192)
#         self.fc3 = nn.Linear(192, n_classes)
#
#     def forward(self, x):
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class MLPRegressionModel_personal(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        activation = nn.Softplus
        self.linear1 = nn.Linear(input_size, input_size)
        self.act1 = activation()
        self.linear2 = nn.Linear(input_size, input_size)
        self.act2 = activation()
        self.linear3 = nn.Linear(input_size, input_size)
        self.act3 = activation()
        self.linear4 = nn.Linear(input_size, 1)

    def forward(self, x):
        h = self.linear1(x)
        h = self.act1(h)

        h = self.linear2(h)
        h = self.act2(h)

        h = self.linear3(h)
        h = self.act3(h)

        h = self.linear4(h)
        return h


class MultivariateMu_personal(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(2, dtype=torch.float32))

    def forward(self, x):
        return self.mu
