import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm.auto import trange
import optuna

from src.utils import compute_metric, compute_likelihood, compute_all_metrics, log_results


def FedSOUL(outer_iters: int,
            inner_iters: int,
            clients_sample_size: int,
            shared_model: nn.Module,
            shared_optim: torch.optim.Optimizer,
            shared_scheduler: torch.optim.lr_scheduler._LRScheduler,
            personal_models: list,
            personal_optims: list,
            personal_schedulers: list,
            backbone_optims: list,
            backbone_schedulers: list,
            prior_models: list,
            prior_optims: list,
            prior_schedulers: list,
            local_dataloaders: list,
            log_likelihood_fn: callable,
            burn_in: int = 0,
            device: str = 'cpu',
            composition_regime: str = 'concatenation',
            use_sgld: bool = False,
            verbose: bool = False,
            verbose_freq: int = 5,
            test_loaders: Optional[list] = None,
            metrics: Optional[list] = None,
            trial: Optional[object] = None,
            ) -> list:
    """
    The function implements Algorithm S1 from the supplementary.
    Parameters
    ----------
    outer_iters: int,
        Number of outer iterations (K from the algorithm)
    inner_iters: int,
        Number of inner iterations (M from the algorithm)
    clients_sample_size: int,
        Number active clients, which will be subsampled during the training
    shared_model: nn.Module,
        Shared model class
    personal_models: list,
        List of PersonalModel (Local) classes.
    prior_models: list,
        List of subclasses of BasePriorModel.
    local_dataloaders: list,
        List of local dataloaders
    log_likelihood_fn: callable,
        Log likelihood computation function
    burn_in: int = 0,
        Index, starting from which states will be recorded. By default, we collect states from the very beginning
    device: str = 'cpu',
        Device, used for computations
    composition_regime: str = 'concatenation',
        The way how we join predictions of shared model, personal model backbone and personal model itself.
    optim_params: dict = {},
        Dictionary with parameters of optimizers, schedulers.
    use_sgld: bool = False,
        If True, for likelihood computation only one batch will be used.
    verbose: bool,
        The flag indicating whether we need to compute metrics after each epoch or no.
    verbose_freq: int,
        How often we calculate the metrics.
    test_loaders: Optional[list]
        The list of test_loaders
    metrics: Optional[list]
        The list of metrics, used for evaluation.
    trial:
        Instance of Optune
    Returns
    -------
        List of trained local models.
    """
    best_test_metric = -float("inf")
    shared_prior_model = len(prior_models) == 1  # True -- prior model same for all clients; False --  personalized

    for outer_iter in trange(outer_iters, desc="Cycle over outer iterations"):  # in the notation of alg, cycle over K
        active_clients_indices = np.random.choice(np.arange(len(personal_models)), size=clients_sample_size,
                                                  replace=False)  # choose of active clients subsample
        shared_optim.zero_grad()
        shared_model_loss = []
        for j in active_clients_indices:  # TODO: Make this cycle in parallel
            if use_sgld:
                multiplier = (local_dataloaders[j].dataset.dataset_size / local_dataloaders[j].batch_size)
            else:
                multiplier = 1.
            pr_index = 0 if shared_prior_model else j  # if prior model is shared, we always use index of 0. Else, j.
            approximate_theta_samples = {"with_ll": torch.tensor([], device=device),
                                         "without_ll": torch.tensor([], device=device)}

            personal_optims[j].zero_grad()  # to make sure cumulative gradients are indeed zeros
            prior_optims[pr_index].zero_grad()  # to make sure cumulative gradients are indeed zeros

            for p in shared_model.parameters():
                p.requires_grad_(False)
            for p in personal_models[j].model.parameters():
                p.requires_grad_(True)
            for p in personal_models[j].backbone_model.parameters():
                p.requires_grad_(True)
            for p in prior_models[pr_index].parameters():
                p.requires_grad_(False)

            shared_model.eval()
            personal_models[j].train()

            # if the model is easy enough to estimate grad log prob, we do not sample theta
            if prior_models[pr_index].sample_theta:
                list_of_sampling_options = ["with_ll", "without_ll"]
            else:
                list_of_sampling_options = ["with_ll"]  # if the model is easy

            # compact way to compute SGLD with and without likelihood term
            for compute_likelihood_term in list_of_sampling_options:
                vector_to_parameters(
                    vec=personal_models[j].theta[compute_likelihood_term],
                    parameters=personal_models[j].model.parameters()
                )
                mean_personal_loss = 0.
                for m in range(inner_iters):  # Computation of key quantities using Langevin
                    prior_negative_log_prob = prior_models[pr_index].get_negative_log_prob(
                        parameters_to_vector(personal_models[j].model.parameters()))
                    log_likelihood_term = torch.zeros_like(prior_negative_log_prob)
                    if compute_likelihood_term == "with_ll":
                        log_likelihood_term = compute_likelihood(personal_model=personal_models[j],
                                                                 shared_model=shared_model,
                                                                 dataloader=local_dataloaders[j],
                                                                 log_likelihood_fn=log_likelihood_fn,
                                                                 composition_regime=composition_regime,
                                                                 use_sgld=use_sgld)

                    personal_model_loss = prior_negative_log_prob - multiplier * log_likelihood_term
                    personal_model_loss.backward()
                    mean_personal_loss += personal_model_loss.item()

                    personal_optims[j].step()
                    personal_schedulers[j].step()
                    personal_optims[j].zero_grad()

                    if personal_models[j].backbone_n_parameters > 0:
                        backbone_optims[j].step()
                        backbone_schedulers[j].step()
                        backbone_optims[j].zero_grad()

                    if m >= burn_in:
                        current_sample = parameters_to_vector(personal_models[j].model.parameters())[None].detach()

                        approximate_theta_samples[compute_likelihood_term] = torch.cat(
                            [approximate_theta_samples[compute_likelihood_term], current_sample])

                personal_models[j].theta[compute_likelihood_term] = current_sample[0]  # 0 here to unsqueeze dimension

            # if verbose and (outer_iter % verbose_freq == 0):
            if j in [0, 50, 99]:
                print(f"Personal model {j} mean loss is {mean_personal_loss / inner_iters}")
            theta = approximate_theta_samples["with_ll"]
            if prior_models[pr_index].sample_theta:
                theta_tilde = approximate_theta_samples["without_ll"]

            for p in shared_model.parameters():
                p.requires_grad_(False)
            for p in personal_models[j].model.parameters():
                p.requires_grad_(False)
            for p in personal_models[j].backbone_model.parameters():
                p.requires_grad_(False)
            for p in prior_models[pr_index].parameters():
                p.requires_grad_(True)

            personal_models[j].eval()
            shared_model.eval()

            I = torch.mean(prior_models[pr_index].get_negative_log_prob(theta), dim=0)
            if prior_models[pr_index].sample_theta:
                I_tilde = torch.mean(prior_models[pr_index].get_negative_log_prob(theta_tilde), dim=0)
            else:
                I_tilde = torch.zeros_like(I)

            if shared_prior_model:
                (-(len(personal_models) / clients_sample_size) * (I_tilde - I)).backward()
            else:
                ## Update energy model parameters
                sum_log_prior = prior_models[pr_index].log_prior_prob()
                accumulation = -(sum_log_prior - I + I_tilde)
                accumulation.backward()
                prior_optims[pr_index].step()
                prior_optims[pr_index].zero_grad()
                prior_schedulers[pr_index].step()

            # personal_optims[j].zero_grad()  # redundant?

            for p in shared_model.parameters():
                p.requires_grad_(True)
            for p in prior_models[pr_index].parameters():
                p.requires_grad_(False)

            shared_model.train()

            ## Phi machinery...
            for phi_iter in range(
                    inner_iters - burn_in):  # range(inner_iters - burn_in):  # worth to introduce another variable
                accumulated_ll = None
                for personal_vector in theta:
                    vector_to_parameters(vec=personal_vector,
                                         parameters=personal_models[j].model.parameters())
                    for p in personal_models[j].model.parameters():  # check if it is needed!
                        p.requires_grad_(False)
                    for p in personal_models[j].backbone_model.parameters():
                        p.requires_grad_(False)
                    current_ll = compute_likelihood(personal_model=personal_models[j],
                                                    shared_model=shared_model,
                                                    dataloader=local_dataloaders[j],
                                                    log_likelihood_fn=log_likelihood_fn,
                                                    composition_regime=composition_regime,
                                                    use_sgld=use_sgld)[None]

                    if accumulated_ll is None:
                        accumulated_ll = current_ll
                    else:
                        accumulated_ll = torch.cat([accumulated_ll, multiplier * current_ll])

                accumulated_ll = torch.mean(accumulated_ll, dim=0)

                (-(len(personal_models) / clients_sample_size) * accumulated_ll).backward()
                shared_optim.step()
                shared_scheduler.step()
                shared_optim.zero_grad()

                shared_model_loss.append((-(len(personal_models) / clients_sample_size) * accumulated_ll).item())

        shared_model_loss = np.mean(shared_model_loss)
        print(f"Epoch {outer_iter}, shared_model_loss is {shared_model_loss}")

        for p in shared_model.parameters():
            p.requires_grad_(False)

        shared_model.eval()

        if shared_prior_model:
            for p in prior_models[0].parameters():
                p.requires_grad_(True)
            ## Update energy model parameters
            sum_log_prior = prior_models[0].log_prior_prob()
            (-sum_log_prior).backward()
            prior_optims[0].step()
            prior_optims[0].zero_grad()
            prior_schedulers[0].step()
            for p in prior_models[0].parameters():
                p.requires_grad_(False)

        if verbose and outer_iter % verbose_freq == 0:
            _, current_test_metric = compute_all_metrics(metrics=metrics, outer_iter=outer_iter,
                                                         test_loaders=test_loaders, local_dataloaders=local_dataloaders,
                                                         composition_regime=composition_regime,
                                                         shared_model=shared_model,
                                                         personal_models=personal_models)
            if current_test_metric > best_test_metric:
                best_shared_model = copy.deepcopy(shared_model)
                best_personal_models = copy.deepcopy(personal_models)
                best_prior_models = copy.deepcopy(prior_models)
                best_test_metric = current_test_metric
            if trial is not None:
                trial.report(current_test_metric, outer_iter)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    if verbose:
        return [best_shared_model, best_personal_models, best_prior_models]
    else:
        return [shared_model, personal_models, prior_models]
