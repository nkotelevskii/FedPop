import torch.nn as nn
import torch.nn.functional as F


class BasePersonalBackbone(nn.Module):
    def __init__(self, backbone_embedding_size, **kwargs):
        super().__init__()
        self.backbone_embedding_size = backbone_embedding_size


class IdenticalBackbone(BasePersonalBackbone):
    def forward(self, x):
        return x


class IdenticalMLPBackbone(BasePersonalBackbone):
    def __init__(self, backbone_embedding_size, input_size):
        super().__init__(backbone_embedding_size=backbone_embedding_size)
        self.net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.ReLU(),
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.ReLU(),
            nn.Linear(in_features=input_size, out_features=input_size),
        )

    def forward(self, x):
        return self.net(x)


class LeNet5Backbone(nn.Module):
    """
    LeNet5 for MNIST
    """

    def __init__(self, backbone_embedding_size):
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
        self.fc3 = nn.Linear(84, backbone_embedding_size)
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


class CNNBackbone(nn.Module):
    """
    CNN from McMahan et al., 2016
    """

    def __init__(self, backbone_embedding_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, backbone_embedding_size)

    def forward(self, x):
        y = self.conv1(x)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        return y


class BigBaseConvNetBackbone(nn.Module):
    def __init__(self, backbone_embedding_size):
        super(BigBaseConvNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)  # nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(1024, backbone_embedding_size)  # nn.Linear(1600, backbone_embedding_size)
        # self.fc1 = nn.LazyLinear(backbone_embedding_size)
        # self.fc2 = nn.Linear(384, 192)
        # self.fc3 = nn.Linear(192, backbone_embedding_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
