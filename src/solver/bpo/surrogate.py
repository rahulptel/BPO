import time

from botorch import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


class Surrogate:
    def __init__(self, config=None):
        self.config = config

    def fit(self, x=None, y=None, time_dict=None):
        raise NotImplementedError


class GPSurrogate(Surrogate):
    def __init__(self, config=None):
        super().__init__(config)

    def _build_gp_components(self, x, y):
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])

        models = []
        for i in range(y_flat.shape[-1]):
            y_flat_obj = y_flat[:, i].unsqueeze(-1)
            kernel = None
            if self.config.kernel == "matern":
                matern_nu = (
                    self.config.matern.nu
                    if self.config.matern and self.config.matern.nu
                    else 2.5
                )
                kernel = ScaleKernel(MaternKernel(nu=matern_nu))

            models.append(
                SingleTaskGP(
                    x_flat,
                    y_flat_obj,
                    covar_module=kernel,
                    outcome_transform=Standardize(m=1),
                )
            )

        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return model, mll

    def fit(self, x, y, time_dict):
        t0 = time.time()

        model, mll = self._build_gp_components(x, y)
        fit_gpytorch_mll(mll)

        if "surrogate_training" not in time_dict:
            time_dict["surrogate_training"] = time.time() - t0
        else:
            time_dict["surrogate_training"] += time.time() - t0

        return model


class IBNNSurrogate(Surrogate):
    def __init__(self, config=None):
        super().__init__(config)

    def fit(self, x, y, time_dict):
        raise NotImplementedError("IBNN surrogate is not implemented yet.")


SURROGATE_REGISTRY = {
    "gp": GPSurrogate,
    "ibnn": IBNNSurrogate,
}


def build_surrogate(config):
    key = config.name.lower()
    if key not in SURROGATE_REGISTRY:
        available = ", ".join(sorted(SURROGATE_REGISTRY))
        raise ValueError(f"Unknown surrogate '{key}'. Available: {available}")
    return SURROGATE_REGISTRY[key](config)


def available_surrogates():
    return tuple(SURROGATE_REGISTRY.keys())
