import numpy as np
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from torch.optim import Optimizer

from ..utils import compute_likelihood


def optimization_factory(optimization_params, parameters):
    if optimization_params['optimizer'] not in ['SGHMC', 'SGLD']:
        optimizer = getattr(torch.optim, optimization_params['optimizer'])(params=parameters,
                                                                           **optimization_params['optimizer_params'])
    elif optimization_params['optimizer'] == 'SGHMC':
        optimizer = SGHMC(params=parameters, **optimization_params['optimizer_params'])
    elif optimization_params['optimizer'] == 'SGLD':
        optimizer = SGLD(params=parameters, **optimization_params['optimizer_params'])
    else:
        raise ValueError(
            "Such optimizer is not available! You can use only standard torch optimizers or SGLD/SGHMC")

    scheduler = getattr(torch.optim.lr_scheduler, optimization_params['scheduler'])(optimizer=optimizer,
                                                                                    **optimization_params[
                                                                                        'scheduler_params'])
    return optimizer, scheduler


class ULA:
    """
    Unadjusted Langevin MCMC
    """

    def __init__(self, step_size: float):
        """

        Parameters
        ----------
        step_size: float
            The step size of a single ULA transition
        """
        self.step_size = step_size

    def forward(self, current_state: Tensor,
                energy_model: torch.nn.Module,
                n_transitions: int,
                whole_model: torch.nn.Module = None,
                local_dataloader: torch.utils.data.DataLoader = None,
                compute_likelihood_term: bool = False,
                return_all_samples: bool = False,
                log_likelihood_fn: callable = nn.CrossEntropyLoss(),
                n_personal_layers: int = 1,
                burn_in: int = 0,
                ) -> Tensor:
        """
        The function performs n_transitions forward steps of ULA
        Parameters
        ----------
        current_state: Tensor
            It is a PyTorch tensor, which represents the current state
        energy_model: torch.nn.Module,
            The model, which guids the sampling procedure
        n_transitions: int
            Number of ULA transitions
        whole_model: torch.nn.Module or None
            If we also need a likelihood function to compute ULA transitions, we should provide the whole model
        local_dataloader: torch.utils.data.DataLoader or None
            In case we need a likelihood function to compute ULA transitions, we should provide the local dataloader
        compute_likelihood_term: bool = False,
            If True, we will compute the likelihood term. Otherwise we do not compute it.
        return_all_samples: bool = False,
            If True, we return all samples along the chain. Otherwise return only the last ones.
        log_likelihood_fn: callable = nn.CrossEntropyLoss(),
            Log likelihood computation function
        n_personal_layers: int = 1,
            Number of layers, considered to be local.
        burn_in: int = 0,
            Index, starting from which states will be recorded.
        Returns
        -------
        final_state: Tensor
            PyTorch tensor, which is the final state.
        """
        with torch.no_grad():
            if return_all_samples:
                final_state = torch.clone(current_state)[None]
            for t in range(n_transitions):
                current_state = self.single_ula_transition(energy_model=energy_model,
                                                           current_state=current_state,
                                                           whole_model=whole_model,
                                                           local_dataloader=local_dataloader,
                                                           compute_likelihood_term=compute_likelihood_term,
                                                           log_likelihood_fn=log_likelihood_fn,
                                                           n_personal_layers=n_personal_layers
                                                           )
                if return_all_samples and t >= burn_in:
                    final_state = torch.cat([final_state, current_state[None]])
            if not return_all_samples:
                final_state = current_state
            else:
                final_state = final_state[1:]
            return final_state

    def single_ula_transition(self, energy_model, current_state, whole_model, local_dataloader,
                              compute_likelihood_term, log_likelihood_fn, n_personal_layers):
        with torch.autograd.enable_grad():
            current_state.requires_grad_(True)
            energy = energy_model(current_state)
            log_likelihood_grad = torch.zeros_like(energy)
            if compute_likelihood_term:
                whole_model.update_personalized_layers(parameters=current_state, n_personal_layers=n_personal_layers)
                log_likelihood_term = compute_likelihood(model=whole_model, dataloader=local_dataloader,
                                                         log_likelihood_fn=log_likelihood_fn)
                log_likelihood_term.backward()
                log_likelihood_grad = whole_model.get_personalized_layers_grad(n_personal_layers=n_personal_layers)
                whole_model.set_grads_to_zero()

            energy_grad = torch.autograd.grad(outputs=energy,
                                              inputs=current_state, only_inputs=True)[0]
            cumulative_grad = energy_grad - log_likelihood_grad
            current_state = current_state.detach()

        new_state = current_state - self.step_size * cumulative_grad + torch.randn_like(cumulative_grad) * np.sqrt(
            2. * self.step_size)
        return new_state


class SGLD(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics with noise rescaled in eaach dimension
        according to RMSProp.
    """

    def __init__(self,
                 params,
                 lr=1e-2,
                 precondition_decay_rate=0.95,
                 num_pseudo_batches=1,
                 num_burn_in_steps=3000,
                 diagonal_bias=1e-8) -> None:
        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        precondition_decay_rate : float, optional
            Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
            Default: `0.95`
        num_pseudo_batches : int, optional
            Effective number of minibatches in the data set.
            Trades off noise and prior with the SGD likelihood term.
            Note: Assumes loss is taken as mean over a minibatch.
            Otherwise, if the sum was taken, divide this number by the batch size.
            Default: `1`.
        num_burn_in_steps : int, optional
            Number of iterations to collect gradient statistics to update the
            preconditioner before starting to draw noisy samples.
            Default: `3000`.
        diagonal_bias : float, optional
            Term added to the diagonal of the preconditioner to prevent it from
            degenerating.
            Default: `1e-8`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, precondition_decay_rate=precondition_decay_rate,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=num_burn_in_steps,
            diagonal_bias=float(diagonal_bias),
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                num_pseudo_batches = group["num_pseudo_batches"]
                precondition_decay_rate = group["precondition_decay_rate"]
                gradient = parameter.grad.data

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.zeros_like(parameter)
                    # state["momentum"] = torch.ones_like(parameter)

                #  }}} State initialization #

                state["iteration"] += 1

                momentum = state["momentum"]

                #  Momentum update {{{ #
                momentum.add_(
                    (1.0 - precondition_decay_rate) * ((gradient ** 2) - momentum)
                )
                #  }}} Momentum update #

                if state["iteration"] > group["num_burn_in_steps"]:
                    sigma = 1. / torch.sqrt(torch.tensor(lr))
                else:
                    sigma = torch.zeros_like(parameter)

                preconditioner = (
                        1. / torch.sqrt(momentum + group["diagonal_bias"])
                )

                scaled_grad = (
                        0.5 * preconditioner * gradient * num_pseudo_batches +
                        torch.normal(
                            mean=torch.zeros_like(gradient),
                            std=torch.ones_like(gradient)
                        ) * sigma * torch.sqrt(preconditioner)
                )

                parameter.data.add_(-lr * scaled_grad)

        return loss


class SGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
        procedure to adapt its own hyperparameters during the initial stages
        of sampling.

        See [1] for more details on this burn-in procedure.\n
        See [2] for more details on Stochastic Gradient Hamiltonian Monte-Carlo.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            In Advances in Neural Information Processing Systems 29 (2016).\n
            `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_
        [2] T. Chen, E. B. Fox, C. Guestrin
            In Proceedings of Machine Learning Research 32 (2014).\n
            `Stochastic Gradient Hamiltonian Monte Carlo <https://arxiv.org/pdf/1402.4102.pdf>`_
    """
    name = "SGHMC"

    def __init__(self,
                 params,
                 lr: float = 1e-2,
                 num_burn_in_steps: int = 3000,
                 noise: float = 0.,
                 mdecay: float = 0.05,
                 scale_grad: float = 1.) -> None:
        """ Set up a SGHMC Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr: float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        num_burn_in_steps: int, optional
            Number of burn-in steps to perform. In each burn-in step, this
            sampler will adapt its own internal parameters to decrease its error.
            Set to `0` to turn scale adaption off.
            Default: `3000`.
        noise: float, optional
            (Constant) per-parameter noise level.
            Default: `0.`.
        mdecay:float, optional
            (Constant) momentum decay per time-step.
            Default: `0.05`.
        scale_grad: float, optional
            Value that is used to scale the magnitude of the noise used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset.
            Default: `1.0`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            noise=noise
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(parameter)
                    state["g"] = torch.ones_like(parameter)
                    state["v_hat"] = torch.ones_like(parameter)
                    state["momentum"] = torch.zeros_like(parameter)
                #  }}} State initialization #

                state["iteration"] += 1

                #  Readability {{{ #
                mdecay, noise, lr = group["mdecay"], group["noise"], group["lr"]
                scale_grad = torch.tensor(group["scale_grad"])

                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]
                momentum = state["momentum"]

                gradient = parameter.grad.data
                #  }}} Readability #

                r_t = 1. / (tau + 1.)
                minv_t = 1. / torch.sqrt(v_hat)

                #  Burn-in updates {{{ #
                if state["iteration"] <= group["num_burn_in_steps"]:
                    # Update state
                    tau.add_(1. - tau * (g * g / v_hat))
                    g.add_(-g * r_t + r_t * gradient)
                    v_hat.add_(-v_hat * r_t + r_t * (gradient ** 2))
                #  }}} Burn-in updates #

                lr_scaled = lr / torch.sqrt(scale_grad)

                #  Draw random sample {{{ #

                noise_scale = (
                        2. * (lr_scaled ** 2) * mdecay * minv_t -
                        2. * (lr_scaled ** 3) * (minv_t ** 2) * noise -
                        (lr_scaled ** 4)
                )

                sigma = torch.sqrt(torch.clamp(noise_scale, min=1e-16))

                # sample_t = torch.normal(mean=0., std=torch.tensor(1.)) * sigma
                sample_t = torch.normal(mean=0., std=sigma)
                #  }}} Draw random sample #

                #  SGHMC Update {{{ #
                momentum_t = momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
                )

                parameter.data.add_(momentum_t)
                #  }}} SGHMC Update #

        return loss
