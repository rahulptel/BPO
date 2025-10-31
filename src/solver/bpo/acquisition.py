import time
from abc import ABC, abstractmethod

import torch
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)


class AcquisitionFunction:
    def __init__(
        self,
        batch_size=None,
        num_restarts=None,
        raw_samples=None,
        sequential=None,
        acq_options=None,
        mc_samples=128,
        n_objs=3,
        ref_point=None,
        rseed=123,
    ):
        self.n_objs = n_objs
        self.batch_size = batch_size
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.sequential = sequential
        self.acq_options = acq_options if acq_options is not None else {}
        self.mc_samples = mc_samples
        self.rseed = rseed

        self.ref_point = torch.tensor(ref_point, dtype=torch.get_default_dtype())
        self.bounds = self.lambda_bounds()
        self.equality_constraints = self.lambda_equality_constraints()

        assert self.ref_point is not None, "Reference point must be provided."

    def lambda_bounds(self):
        lower = torch.zeros(self.n_objs, dtype=torch.get_default_dtype())
        upper = torch.ones(self.n_objs, dtype=torch.get_default_dtype())
        return torch.stack([lower, upper])

    def lambda_equality_constraints(self):
        indices = torch.arange(self.n_objs)
        coeffs = torch.ones(self.n_objs, dtype=torch.get_default_dtype())
        return [(indices, coeffs, 1.0)]

    def generate_candidates(self, model, train_x, time_dict):
        raise NotImplementedError


class QLogEHVIAcquisition(AcquisitionFunction):
    def __init__(
        self,
        batch_size=None,
        num_restarts=None,
        raw_samples=None,
        sequential=None,
        acq_options=None,
        mc_samples=128,
        n_objs=None,
        ref_point=None,
        rseed=123,
    ):
        super().__init__(
            batch_size=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=sequential,
            acq_options=acq_options,
            mc_samples=mc_samples,
            n_objs=n_objs,
            ref_point=ref_point,
            rseed=rseed,
        )
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.mc_samples]),
            seed=self.rseed,
        )

    def generate_candidates(self, model, x, time_dict):
        if model is None:
            raise ValueError("qLogEHVI acquisition requires a surrogate model.")

        t0 = time.time()
        with torch.no_grad():
            posterior_mean = model.posterior(x).mean

        partitioning = FastNondominatedPartitioning(
            ref_point=self.ref_point,
            Y=posterior_mean,
        )
        if "partitioning" not in time_dict:
            time_dict["partitioning"] = time.time() - t0
        else:
            time_dict["partitioning"] += time.time() - t0

        acq_func = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point,
            partitioning=partitioning,
            sampler=self.sampler,
        )
        options = {"maxiter": 200}
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=options,
            sequential=self.sequential,
            bounds=self.bounds,
            equality_constraints=self.equality_constraints,
        )
        if "acquisition_optimization" not in time_dict:
            time_dict["acquisition_optimization"] = time.time() - t0
        else:
            time_dict["acquisition_optimization"] += time.time() - t0

        return candidates


ACQUISITION_REGISTRY = {
    "qlogehvi": QLogEHVIAcquisition,
}


def build_acquisition(config, n_objs=None, ref_point=None, rseed=None):
    key = config.name.lower()
    if key not in ACQUISITION_REGISTRY:
        raise ValueError(f"Unknown acquisition function '{key}'")

    if key == "qlogehvi":
        return QLogEHVIAcquisition(
            batch_size=config.batch_size_q,
            num_restarts=config.num_restarts,
            raw_samples=config.raw_samples,
            sequential=config.sequential,
            mc_samples=config.mc_samples,
            n_objs=n_objs,
            ref_point=ref_point,
            rseed=rseed,
        )


def available_acquisitions():
    return tuple(ACQUISITION_REGISTRY.keys())
