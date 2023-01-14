import os
import typing
from types import ModuleType
from typing import Tuple, Optional

import albumentations as A
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import random

AVAILABLE_DATASETS = ['cifar10', 'cifar100', 'mnist', 'femnist', 'gaussians', 'sinusoid_regression',
                      'multivariate_mean_prediction', 'reproduce_params']


def noniid(dataset, n_clients, classes_per_user, num_classes, rand_set_all=[], testb=False, use_leftover=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param n_clients:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(n_clients)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(classes_per_user * n_clients / num_classes)
    samples_per_user = int(count / n_clients)
    # whether to sample more test samples per user
    if (samples_per_user < 100):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        if use_leftover:
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        else:
            x = np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((n_clients, -1))

    # divide and assign
    for i in range(n_clients):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100 and testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)

    return dict_users, rand_set_all


def get_transforms(dataset_name: str) -> torchvision.transforms:
    """
    The function implements the set of transforms for a given dataset.
    Parameters
    ----------
    dataset_name: str
        Dataset name.

    Returns
    -------
        A set of transformations for a given dataset name.
    """
    transforms_dict = {}
    transforms_dict.update({'cifar10_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])})
    transforms_dict.update({'cifar10_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])})

    transforms_dict.update({'cifar100_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441],
                             std=[0.267, 0.256, 0.276])])})
    transforms_dict.update({'cifar100_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441],
                             std=[0.267, 0.256, 0.276])])})

    transforms_dict.update({'mnist': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])})

    transforms_dict.update({'femnist': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))])})

    if dataset_name not in transforms_dict.keys():
        transforms_dict[dataset_name] = None
    return transforms_dict[dataset_name]


def get_data(dataset_name, **specific_dataset_params):
    dataset_name = dataset_name.lower()
    if dataset_name.lower() not in AVAILABLE_DATASETS:
        raise ValueError(
            f"'{dataset_name}' is not in list of {AVAILABLE_DATASETS=}")
    elif dataset_name == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=get_transforms("mnist"))
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=get_transforms("mnist"))
        # sample users
        dict_users_train, rand_set_all = noniid(dataset_train, **specific_dataset_params)
        dict_users_test, rand_set_all = noniid(dataset_test, **specific_dataset_params, rand_set_all=rand_set_all)
    elif dataset_name == 'femnist':
        dataset_train = datasets.MNIST('data/fashion_mnist/', train=True, download=True,
                                       transform=get_transforms("femnist"))
        dataset_test = datasets.MNIST('data/fashion_mnist/', train=False, download=True,
                                      transform=get_transforms("femnist"))
        # sample users
        dict_users_train, rand_set_all = noniid(dataset_train, **specific_dataset_params)
        dict_users_test, rand_set_all = noniid(dataset_test, **specific_dataset_params, rand_set_all=rand_set_all)

    elif dataset_name == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True,
                                         transform=get_transforms("cifar10_train"))
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True,
                                        transform=get_transforms("cifar10_test"))
        dict_users_train, rand_set_all = noniid(dataset_train, **specific_dataset_params)
        dict_users_test, rand_set_all = noniid(dataset_test, **specific_dataset_params, rand_set_all=rand_set_all)
    elif dataset_name == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True,
                                          transform=get_transforms("cifar100_train"))
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True,
                                         transform=get_transforms("cifar100_test"))
        dict_users_train, rand_set_all = noniid(dataset_train, **specific_dataset_params)
        dict_users_test, rand_set_all = noniid(dataset_test, **specific_dataset_params, rand_set_all=rand_set_all)

    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])

    # for i in dict_users_train.keys():
    #     labels_tr = []
    #     labels_ts = []
    #     for idx in dict_users_train[i]:
    #         labels_tr.append(dataset_train[idx][1])
    #     for idx in dict_users_test[i]:
    #         labels_ts.append(dataset_test[idx][1])
    #
    #     print(np.unique(labels_ts, return_counts=True))
    #     print(np.unique(labels_tr, return_counts=True))

    return dataset_train, dataset_test, dict_users_train, dict_users_test


def load_numpy_data(dataset_name: str, specific_dataset_params: Optional[dict],
                    root_path: str = './data/') -> Tuple:
    """
    The function loads numpy data
    Parameters
    ----------
    dataset_name: str
        Dataset name.
    specific_dataset_params: dict
        Specific parameters for this dataset
    root_path: str
        A path to a folder with data

    Returns
    -------
        Tuple of (X_train, X_test, y_train, y_test) splits of the dataset.
    """
    if dataset_name.lower() not in AVAILABLE_DATASETS:
        raise ValueError(
            f"'{dataset_name}' is not in list of {AVAILABLE_DATASETS=}")

    dataset_name = form_name(dataset_name, specific_dataset_params)
    PATH = os.path.join(root_path, f'{dataset_name}.npz')
    data = np.load(PATH)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for name in data.keys():
        if name.startswith('x_train_'):
            X_train.append(data[name])
            y_train.append(data[f'y_train_{int(name.split("_")[-1])}'])
    if len(X_train) == 0:
        X_train = data["x_train"]
        y_train = data["y_train"]

    for name in data.keys():
        if name.startswith('x_test_'):
            X_test.append(data[name])
            y_test.append(data[f'y_test_{int(name.split("_")[-1])}'])
    if len(X_test) == 0:
        X_test = data["x_test"]
        y_test = data["y_test"]

    return X_train, X_test, y_train, y_test


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray,
                 y: np.ndarray,
                 transforms: typing.Optional[ModuleType] = None,
                 device: str = "cuda",
                 regression: bool = False):
        """
        Custom dataset class
        Parameters
        ----------
        X: np.ndarray
            Objects of the dataset
        y: np.ndarray
            Labels of the dataset
        transforms: torchvision.transforms
            Set of transforms
        device: str
            Specific device data will be sent to
        regression: bool
            Flag, True if we consider regression problem. For classification, we need to cast labels as Long tensor
        """
        self.X = X
        self.y = y
        self.transforms = transforms
        self.device = device
        self.regression = regression
        self.dataset_size = len(self.X)

    def __len__(self, ):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx].astype(np.float32)
        if self.regression:
            y = self.y[idx]
            if len(y.shape) < 2:
                y = y[None]
            y = torch.FloatTensor(y)
        else:
            y = torch.LongTensor(self.y[idx][None])  # .astype(np.float32)
        if self.transforms is not None:
            X = self.transforms(X)
        else:
            X = torch.tensor(X)
        if self.device.find('cuda') > -1:
            X = X.to(self.device)
            y = y.to(self.device)
        return X, y


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None, device='cpu'):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name
        self.device = device
        self.dataset_size = len(self.idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'mnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]), (1, 28, 28))
            label = torch.tensor(self.dataset['y'][item])
        else:
            image, label = self.dataset[self.idxs[item]]
        label = torch.LongTensor(np.array(label)[None])
        return image.to(self.device), label.to(self.device)


def generate_dataloaders(dataset_name: str,
                         specific_dataset_params: dict,
                         batch_size_train: int,
                         batch_size_test: int = None,
                         max_dataset_size_per_user: int = 500,
                         min_dataset_size_per_user: int = 500,
                         n_clients_with_min_datasets: int = 0,
                         DEVICE: str = 'cpu',
                         **kwargs
                         ) -> tuple:
    """
    The function yields training and test dataloaders
    Parameters
    ----------

    dataset_name: str,
        Dataset name.
    specific_dataset_params: dict,
        Specific parameters for this dataset
    batch_size_train: int,
        Batch size of train dataset
    batch_size_test: int,
        Batch size of test dataset
    DEVICE: str,
        Specific device data will be sent to
    regression: bool,
        Flag, if True -- the data will be processed in a way regression data should be prepared
    Returns
    -------
        Tuple of train_loader, test_loader
    """
    print("Loading data...")
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(dataset_name=dataset_name,
                                                                              **specific_dataset_params)
    print("Done!")

    train_loaders = []
    test_loaders = []

    if DEVICE is None:
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for i in dict_users_train.keys():
        if i < n_clients_with_min_datasets:
            train_loaders.append(
                DataLoader(DatasetSplit(dataset_train, dict_users_train[i][:min_dataset_size_per_user], device=DEVICE),
                           batch_size=batch_size_train))
        else:
            train_loaders.append(
                DataLoader(DatasetSplit(dataset_train, dict_users_train[i][:max_dataset_size_per_user], device=DEVICE),
                           batch_size=batch_size_train))

    for i in dict_users_test.keys():
        test_loaders.append(
            DataLoader(DatasetSplit(dataset_test, dict_users_test[i], device=DEVICE),
                       batch_size=batch_size_test))

    return train_loaders, test_loaders


def load_dataloaders(dataset_name: str,
                     specific_dataset_params: dict,
                     root_path: str,
                     batch_size_train: int,
                     batch_size_test: int = None,
                     DEVICE: str = None,
                     regression: bool = False,
                     **kwargs
                     ) -> tuple:
    """
    The function yields training and test dataloaders
    Parameters
    ----------
    dataset_name: str,
        Dataset name.
    specific_dataset_params: dict,
        Specific parameters for this dataset
    root_path: str,
        Path to a folder with dataset of interest.
    batch_size_train: int,
        Batch size of train dataset
    batch_size_test: int,
        Batch size of test dataset
    DEVICE: str,
        Specific device data will be sent to
    regression: bool,
        Flag, if True -- the data will be processed in a way regression data should be prepared
    Returns
    -------
        Tuple of train_loader, test_loader
    """
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_numpy_data(dataset_name=dataset_name,
                                                       specific_dataset_params=specific_dataset_params,
                                                       root_path=root_path)
    print("Done!")
    transforms_train = get_transforms(dataset_name + '_train')
    transforms_test = get_transforms(dataset_name + '_test')

    if DEVICE is None:
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    trainsets = []
    for X, y in zip(X_train, y_train):
        trainsets.append(CustomDataset(X, y, transforms=transforms_train, device=DEVICE, regression=regression))

    batch_size = batch_size_train
    train_loaders = []
    for trainset in trainsets:
        train_loaders.append(DataLoader(trainset, batch_size=batch_size, shuffle=True))

    test_batch_size = batch_size if batch_size_test is None else batch_size_test
    if isinstance(X_test, list):
        testsets = []
        for X, y in zip(X_test, y_test):
            testsets.append(CustomDataset(X, y, transforms=transforms_test, device=DEVICE, regression=regression))
        test_loader = []
        for testset in testsets:
            test_loader.append(DataLoader(testset, batch_size=test_batch_size, shuffle=False))
    else:
        testset = CustomDataset(X_test, y_test, transforms=transforms_test, device=DEVICE, regression=regression)
        test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    return train_loaders, test_loader


def get_dataloaders(generate_data, **kwargs) -> tuple:
    if generate_data:
        return generate_dataloaders(**kwargs)
    else:
        return load_dataloaders(**kwargs)


def form_name(dataset_name, kwargs):
    if dataset_name.find('gaussians') > -1:
        return str(kwargs['num_classes']) + '_' + 'gaussians' + '_Uniform_' + str(kwargs['uniform'])
    elif dataset_name.find('sinusoid_regression') > -1:
        return f"sinusoid_regression_n_clients={kwargs['n_clients']}_distributes_train_size={kwargs['train_size']}"
    elif dataset_name.find('multivariate_mean_prediction') > -1:
        return str(kwargs['n_modes']) + '_multivariate_mean_prediction_' + str(kwargs['n_objects_per_mode_train'])
    elif dataset_name.find('mnist') > -1 or dataset_name.find('cifar10') > -1:
        if kwargs.get('raw_version', False):
            return "numpy_" + dataset_name
        else:
            return f"n_clients={kwargs['n_clients']}_{dataset_name}_pathological_niid"
    elif dataset_name.find('reproduce') > -1:
        return f"n_models={kwargs['n_clients']}_reproduce_params"
