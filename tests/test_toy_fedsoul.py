import sys

sys.path.insert(0, './')

from main import run
from src.utils.utils import get_config, log_results


def test_regression():
    config = get_config(path='configs/toy_regression.yml')
    config['train_params']['seeds'] = [42]
    output = run(config=config)
    metric = output['metrics']['mse'][0]
    # log_results(config_path='configs/toy_regression.yml', config=config, output=output)
    assert metric < 0.1


def test_classification():
    config = get_config(path='configs/toy_classification.yml')
    config['train_params']['seeds'] = [42]
    output = run(config=config)
    metric = output['metrics']['accuracy'][0]
    # log_results(config_path='configs/toy_classification.yml', config=config, output=output)
    assert metric > 0.99


def test_multinomial_mean_prediction():
    config = get_config(path='configs/toy_multivariate_gaussian_mean_prediction.yml')
    config['train_params']['seeds'] = [42]
    output = run(config=config)
    metric = output['metrics']['multivariate_mse'][0]
    # log_results(config_path='configs/toy_multivariate_gaussian_mean_prediction.yml', config=config, output=output)
    assert metric < 0.1
