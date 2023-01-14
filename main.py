from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.fedsoul import FedSOUL
from src.models import PersonalModel
from src.models.model_factory import set_model
from src.samplers import optimization_factory
from src.utils import seed_everything, parse_args, get_config, dotdict, count_parameters, compute_metric, log_results


def run(config, trial=None) -> dict:
    print(config)

    # Here we receive dotdicts, to access fields via dot operator.
    data_params = dotdict(config['data_params'])
    train_params = dotdict(config['train_params'])
    model_params = dotdict(config['model_params'])
    eval_params = dotdict(config['eval_params'])
    optimization_params = dotdict(config['optimization'])

    train_params.n_personal_models = data_params.specific_dataset_params["n_clients"]

    OUTER_ITERS = train_params.outer_iters
    INNER_ITERS = train_params.inner_iters
    DEVICE = train_params.device
    N_PERSONAL_MODELS = train_params.n_personal_models

    # Here we define log prior function
    if train_params.prior:
        LOG_PRIOR_FN = torch.distributions.Normal(loc=torch.tensor(0., dtype=torch.float32, device=DEVICE),
                                                  scale=torch.tensor(1., dtype=torch.float32, device=DEVICE)).log_prob
    else:
        raise ValueError(f"No such prior available! '{train_params.prior}'")

    # Here we define log likelihood function
    if train_params.loss_fn_name == 'mse':
        LOG_LIKELIHOOD_FN = lambda x, y: -nn.MSELoss(reduction='sum')(x.squeeze(), y.squeeze())
    elif train_params.loss_fn_name == 'cross_entropy':
        LOG_LIKELIHOOD_FN = lambda x, y: -nn.CrossEntropyLoss(reduction='sum')(x, y)
    elif train_params.loss_fn_name == 'multivariate_gaussian_ll':
        def multivariate_gaussian_ll(y_pred, y):
            mu = y_pred[:2]
            lower_diag = y_pred[2:].view(2, 2)
            return torch.distributions.MultivariateNormal(loc=mu, scale_tril=lower_diag).log_prob(y).sum()

        LOG_LIKELIHOOD_FN = multivariate_gaussian_ll
    else:
        raise ValueError(f"Wrong loss name...{train_params.loss}")

    #######################################################################################
    ############################# Below, we define models #################################
    #######################################################################################

    # To define personal model (parameterized by \theta), we should decide which input size it takes
    personal_model_params = model_params.personal_model_params
    if model_params.composition_model_regime == "composition":
        personal_model_params['input_size'] = model_params.shared_model_params["shared_embedding_size"]
    else:
        personal_model_params['input_size'] = model_params.shared_model_params["shared_embedding_size"] + \
                                              model_params.backbone_model_params["backbone_embedding_size"]

    metrics_dict = defaultdict(list)

    # Here we start cycle over different seeds
    for seed in train_params.seeds:
        seed_everything(seed)
        train_loaders, test_loaders = get_dataloaders(generate_data=data_params.generate_dataloaders,
                                                      dataset_name=data_params.dataset_name,
                                                      specific_dataset_params=data_params.specific_dataset_params,
                                                      root_path=data_params.root_path,
                                                      batch_size_train=data_params.train_batch_size,
                                                      batch_size_test=data_params.test_batch_size,
                                                      DEVICE=DEVICE,
                                                      regression=data_params.regression,
                                                      max_dataset_size_per_user=data_params.max_dataset_size_per_user,
                                                      min_dataset_size_per_user=data_params.min_dataset_size_per_user,
                                                      n_clients_with_min_datasets=data_params.n_clients_with_min_datasets,
                                                      )

        # We define one shared model
        shared_model = set_model(model_name=model_params.shared_model_name, device=DEVICE,
                                 model_params=model_params.shared_model_params, model_type='shared')
        shared_model.set_log_prior_fn(log_prior_fn=LOG_PRIOR_FN)
        shared_optim, shared_scheduler = optimization_factory(parameters=shared_model.parameters(),
                                                              optimization_params={k[len('shared') + 1:]: v for k, v in
                                                                                   optimization_params.items() if
                                                                                   k.startswith('shared')})

        personal_models = []
        personal_optims = []
        personal_schedulers = []

        backbone_optims = []
        backbone_schedulers = []

        prior_models = []
        prior_optims = []
        prior_schedulers = []

        # And we define N_PERSONAL_MODELS personal models
        for i in range(N_PERSONAL_MODELS):
            #####################################
            #####################################
            # First -- backbone
            backbone_model = set_model(model_name=model_params.backbone_model_name, device=DEVICE,
                                       model_params=model_params.backbone_model_params,
                                       model_type='backbone')
            # Second -- high level personal model
            personal_model = set_model(model_name=model_params.personal_model_name, device=DEVICE,
                                       model_params=personal_model_params, model_type='personal')
            # Energy model takes as an input both vectors -- parameters of personal and personal backbone models
            model_params.prior_model_params.update({
                "in_features": count_parameters(personal_model),
                "log_prior_fn": LOG_PRIOR_FN,
            })
            # An instance of Personal Model consists of sets of parameters \theta, \theta_b
            personal_models.append(PersonalModel(
                model=personal_model,
                backbone_model=backbone_model))

            # And next, we define corresponding optims
            # For personal model
            personal_optim, personal_scheduler = optimization_factory(
                parameters=list(personal_models[i].model.parameters()),
                optimization_params={k[len("personal") + 1:]: v for k, v in
                                     optimization_params.items() if
                                     k.startswith('personal')})
            personal_optims.append(personal_optim)
            personal_schedulers.append(personal_scheduler)

            # And for the backbone
            if personal_models[i].backbone_n_parameters > 0:
                backbone_optim, backbone_scheduler = optimization_factory(
                    parameters=list(personal_models[i].backbone_model.parameters()),
                    optimization_params={k[len("backbone") + 1:]: v for k, v in
                                         optimization_params.items() if
                                         k.startswith('backbone')})
                backbone_optims.append(backbone_optim)
                backbone_schedulers.append(backbone_scheduler)

            #####################################
            #####################################

            # We define \beta (prior model) as an instance of another special class
            if model_params.shared_prior_model and i == 0:  # if we share prior model, than we add it only once
                prior_models.append(set_model(model_name=model_params.prior_model_name, device=DEVICE,
                                              model_params=model_params.prior_model_params, model_type='prior').to(
                    DEVICE))
                prior_optim, prior_scheduler = optimization_factory(
                    parameters=prior_models[0].parameters(),
                    optimization_params={k[len('prior_model') + 1:]: v for k, v in
                                         optimization_params.items() if
                                         k.startswith('prior_model')})
                prior_optims.append(prior_optim)
                prior_schedulers.append(prior_scheduler)
            else:  # else, we have an array of models
                if model_params.shared_prior_model:
                    continue
                prior_models.append(set_model(model_name=model_params.prior_model_name, device=DEVICE,
                                              model_params=model_params.prior_model_params, model_type='prior').to(
                    DEVICE))
                prior_optim, prior_scheduler = optimization_factory(
                    parameters=prior_models[i].parameters(),
                    optimization_params={k[len('prior_model') + 1:]: v for k, v in
                                         optimization_params.items() if
                                         k.startswith('prior_model')})
                prior_optims.append(prior_optim)
                prior_schedulers.append(prior_scheduler)

        ###########################################################################
        ###########################################################################
        initial_metrics = defaultdict(list)
        for metric_name in eval_params.metrics:
            print(f'Checking {metric_name} of initial models')
            m = compute_metric(metric=metric_name, personal_models=personal_models,
                               loaders=test_loaders if isinstance(test_loaders, list) else train_loaders,
                               shared_model=shared_model, composition_regime=model_params.composition_model_regime)
            initial_metrics[metric_name].append(m)

        print('Initial metrics:')
        for k, v in initial_metrics.items():
            print(f"{k} : {np.mean(v)} +/- {0 if len(v) == 0 else np.std(v)}")

        models = FedSOUL(
            outer_iters=OUTER_ITERS,
            inner_iters=INNER_ITERS,
            clients_sample_size=train_params.clients_sample_size,
            personal_models=personal_models,
            personal_optims=personal_optims,
            personal_schedulers=personal_schedulers,
            backbone_optims=backbone_optims,
            backbone_schedulers=backbone_schedulers,
            prior_models=prior_models,
            prior_optims=prior_optims,
            prior_schedulers=prior_schedulers,
            shared_model=shared_model,
            shared_optim=shared_optim,
            shared_scheduler=shared_scheduler,
            local_dataloaders=train_loaders,
            log_likelihood_fn=LOG_LIKELIHOOD_FN,
            burn_in=train_params.inner_burn_in,
            device=DEVICE,
            composition_regime=model_params.composition_model_regime,
            use_sgld=train_params.use_sgld,
            verbose=train_params.verbose,
            verbose_freq=train_params.verbose_freq,
            test_loaders=test_loaders,
            metrics=eval_params.metrics,
            trial=trial,
        )
        shared_model, personal_models, prior_models = models[0], models[1], models[2]

        for metric_name in eval_params.metrics:
            print(f'Checking {metric_name} of final models')
            m = compute_metric(metric=metric_name, personal_models=personal_models,
                               loaders=test_loaders if isinstance(test_loaders, list) else train_loaders,
                               shared_model=shared_model, composition_regime=model_params.composition_model_regime)
            metrics_dict[metric_name].append(m)

        print('Final metrics:')
        for k, v in metrics_dict.items():
            print(f"{k} : {np.mean(v)} +/- {0 if len(v) == 0 else np.std(v)}")

    return {
        "metrics": {k: [np.mean(v), 0 if len(v) == 0 else np.std(v)] for k, v in metrics_dict.items()},
        "shared_model": shared_model,
        "personal_models": personal_models,
        "prior_models": prior_models,
        "train_loaders": train_loaders,
        "test_loaders": test_loaders
    }


if __name__ == '__main__':
    args = parse_args()
    conf = get_config(args.config)
    output = run(conf)
    log_results(config_path=args.config, config=conf, output=output)
