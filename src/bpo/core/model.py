import time

from botorch import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


class Surrogate:
    def __init__(self, config=None):
        self.config = config

    def fit(self, train_x, train_obj):
        raise NotImplementedError


class GPSurrogate(Surrogate):
    @staticmethod
    def _build_gp_components(train_x, train_obj):
        train_x_flat = train_x.reshape(-1, train_x.shape[-1])
        train_obj_flat = train_obj.reshape(-1, train_obj.shape[-1])

        models = []
        for i in range(train_obj_flat.shape[-1]):
            train_y = train_obj_flat[:, i].unsqueeze(-1)
            models.append(
                SingleTaskGP(train_x_flat, train_y, outcome_transform=Standardize(m=1))
            )

        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return model, mll

    def fit(self, train_x, train_obj, time_dict):
        t0 = time.time()

        model, mll = self._build_gp_components(train_x, train_obj)
        fit_gpytorch_mll(mll)

        if "surrogate_training" not in time_dict:
            time_dict["surrogate_training"] = time.time() - t0
        else:
            time_dict["surrogate_training"] += time.time() - t0

        return model


class IBNNSurrogate(Surrogate):
    def fit(self, train_x, train_obj, time_dict):
        raise NotImplementedError("IBNN surrogate is not implemented yet.")


class NoneSurrogate(Surrogate):
    def fit(self, train_x, train_obj, time_dict):
        return None


SURROGATE_REGISTRY = {
    "gp": GPSurrogate,
    "ibnn": IBNNSurrogate,
    "none": NoneSurrogate,
}


def build_surrogate(name, config=None):
    key = name.lower()
    if key not in SURROGATE_REGISTRY:
        available = ", ".join(sorted(SURROGATE_REGISTRY))
        raise ValueError(f"Unknown surrogate '{name}'. Available: {available}")
    return SURROGATE_REGISTRY[key](config)


def available_surrogates():
    return tuple(SURROGATE_REGISTRY.keys())
