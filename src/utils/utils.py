import argparse
import os
import pickle
import random

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score


def pretty_matplotlib_config(fontsize=15):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams.update({'font.size': fontsize})


def compute_likelihood(personal_model: nn.Module, dataloader: torch.utils.data.DataLoader,
                       log_likelihood_fn: callable, shared_model: nn.Module = None,
                       composition_regime: str = 'concatenation', use_sgld=False) -> torch.Tensor:
    """
    The function computes cumulative likelihood for a given dataloader
    Parameters
    ----------
    shared_model: nn.Module
        Shared model
    personal_model: nn.Module
        Model, used for likelihood estimation
    dataloader: torch.utils.data.DataLoader
        Dataloader (usually local) used for likelihood estimation
    log_likelihood_fn: callable
        Log likelihood computation function
    composition_regime: str
        The way how we join predictions of shared model, personal model backbone and personal model itself.
    use_sgld: bool = False,
        If True, for likelihood computation only one batch will be used.
    Returns
    -------
        likelihood: torch.Tensor
            Cumulative likelihood
    """
    log_likelihood = None
    personal_model = composed_model(shared_model=shared_model, personal_model=personal_model,
                                    composition_regime=composition_regime)
    for x, y in dataloader:
        y_pred = personal_model(x)
        y = y.squeeze()
        y_pred = y_pred.squeeze()

        if log_likelihood is None:
            log_likelihood = log_likelihood_fn(y_pred, y)
        else:
            log_likelihood += log_likelihood_fn(y_pred, y)
        if use_sgld:
            break
    return log_likelihood


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_local_params(model, n_personal_layers):
    return count_parameters(
        nn.ModuleList(
            [m for m in [m for m in model.modules()][1:] if isinstance(m, nn.Linear)][-n_personal_layers:]))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default="configs/toy_multivariate_gaussian_mean_prediction.yml")
    # parser.add_argument('--config', type=str, default="configs/toy_regression.yml")
    # parser.add_argument('--config', type=str, default="configs/toy_classification.yml")
    # parser.add_argument('--config', type=str, default="configs/mnist_classification.yml")
    # parser.add_argument('--config', type=str, default="configs/femnist_classification.yml")
    parser.add_argument('--config', type=str, default="configs/cifar10_classification.yml")
    # parser.add_argument('--config', type=str, default="configs/cifar100_classification.yml")
    return parser.parse_args()


def get_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config


def compute_accuracy(model, loader):
    all_predictions = torch.tensor([])
    all_true_labels = torch.tensor([])
    with torch.no_grad():
        for x, y in loader:
            y_pred = torch.argmax(model(x), -1).cpu().squeeze()
            if len(y_pred.shape) == 0:
                y_pred = y_pred.view(1, )
            all_predictions = torch.cat([all_predictions, y_pred])
            y = y.cpu().squeeze()
            if len(y.shape) == 0:
                y = y.view(1, )
            all_true_labels = torch.cat([all_true_labels, y])
    all_predictions = all_predictions.numpy()
    all_true_labels = all_true_labels.numpy()

    accuracy = accuracy_score(all_true_labels, all_predictions)
    return accuracy


def compute_mse(model, loader):
    se = None
    with torch.no_grad():
        for x, y in loader:
            y_pred = model(x).cpu().detach().numpy().squeeze()
            y = y.cpu().detach().squeeze().numpy().squeeze()
            if se is None:
                se = np.array(np.square(y_pred - y))
            else:
                se = np.concatenate([se, np.square(y_pred - y)])

    return np.mean(se)


def compute_multivariate_mse(model, loader):
    true_mu = loader.dataset.X.mean(0)
    true_covariance = np.array([[1., -0.9], [-0.9, 1.]]).reshape(-1)

    with torch.no_grad():
        output = model(torch.empty([]))
        pred_mu = output[:2].cpu().numpy()
        pred_lower = output[2:].view(2, 2)

        pred_sigma = pred_lower @ pred_lower.T
        pred_sigma = pred_sigma.view(-1).cpu().numpy()
    return np.concatenate([np.square(true_mu - pred_mu), np.square(true_covariance - pred_sigma)])


def compute_all_metrics(metrics, personal_models, shared_model, composition_regime, local_dataloaders, test_loaders,
                        outer_iter):
    for metric_name in metrics:
        print(f"{metric_name} after the {outer_iter}'s epoch")
        m_train = compute_metric(metric=metric_name, personal_models=personal_models,
                                 loaders=local_dataloaders,
                                 shared_model=shared_model, composition_regime=composition_regime)
        print(f"Train {metric_name} is {m_train}")
        if isinstance(test_loaders, list):
            m = compute_metric(metric=metric_name, personal_models=personal_models,
                               loaders=test_loaders,
                               shared_model=shared_model, composition_regime=composition_regime)
            print(f"Test {metric_name} is {m}")
    return m_train, m


def compute_metric(metric, personal_models, loaders, composition_regime, shared_model=None):
    metrics = []
    if shared_model is not None:
        shared_model.eval()
    for i in range(len(personal_models)):
        personal_models[i].eval()
        if shared_model is not None:
            personal_model = composed_model(shared_model=shared_model, personal_model=personal_models[i],
                                            composition_regime=composition_regime)
        else:
            personal_model = personal_models[i].model

        if metric == 'accuracy':
            metrics.append(compute_accuracy(model=personal_model, loader=loaders[i]))
        elif metric == 'mse':
            metrics.append(compute_mse(model=personal_model, loader=loaders[i]))
        elif metric == 'multivariate_mse':
            metrics.append(compute_multivariate_mse(model=personal_model, loader=loaders[i]))
        # print(f"For model number {i} {metric} is {metrics[-1]}")
    return np.mean(metrics)


def composed_model(shared_model, personal_model, composition_regime):
    if composition_regime == 'concatenation':
        return lambda x: personal_model.model(torch.cat([shared_model(x), personal_model.backbone_model(x)], dim=-1))
    if composition_regime == 'composition':
        return lambda x: personal_model.model(shared_model(x))
    elif composition_regime == 'separate_params':
        def separated_composition(x):
            shared_output = shared_model(x)
            personal_output = personal_model.model(personal_model.backbone_model(x))
            return torch.cat([personal_output, shared_output], dim=-1)

        return separated_composition
    elif composition_regime == 'only_shared':
        def only_shared_composition(x):
            shared_output = shared_model(x)
            return shared_output

        return only_shared_composition
    else:
        raise ValueError(f'Wrong composition regime! {composition_regime}')


def log_results(config_path, config, output):
    exp_folder = config['train_params']['exp_folder']
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)

    exp_id = len([s for s in os.listdir(exp_folder) if not s.startswith('.')])
    exp_folder_path = os.path.join(exp_folder, str(exp_id))
    try:
        os.mkdir(exp_folder_path)
    except:
        exp_id = max([int(s) for s in os.listdir(exp_folder) if not s.startswith('.')]) + 1
        exp_folder_path = os.path.join(exp_folder, str(exp_id))
        os.mkdir(exp_folder_path)

    # shutil.copyfile(config_path, os.path.join(exp_folder_path, 'config.yml'))
    with open(os.path.join(exp_folder_path, 'config.yml'), 'w') as file:
        config['model_params']['prior_model_params'] = {k: v for k, v in
                                                        config['model_params']['prior_model_params'].items() if
                                                        not callable(v)}  # need to some something more clever
        yaml.dump(config, file)

    mean_metrics = {}
    final_metrics = output['metrics']
    for k, v in final_metrics.items():
        mean, std = v[0], v[1]
        mean_metrics[k] = f"{mean} +/ - {std}"

    with open(os.path.join(exp_folder_path, 'metrics.yml'), 'w') as file:
        yaml.dump(mean_metrics, file)

    try:
        with open(os.path.join(exp_folder_path, 'final_dict.pickle'), 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(f"Final dict is not saved duу to {ex}")
