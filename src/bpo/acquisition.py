import time
from abc import ABC, abstractmethod

import torch
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)


class AcquisitionFunction(ABC):
    def __init__(
        self,
        ref_point=None,
        bounds=None,
        batch_size=None,
        num_restarts=None,
        raw_samples=None,
        sequential=None,
        equality_constraints=None,
        acq_options=None,
        mc_samples=128,
        rseed=123,
    ):
        self.ref_point = ref_point
        self.bounds = bounds
        self.batch_size = batch_size
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.sequential = sequential
        self.equality_constraints = equality_constraints
        self.acq_options = acq_options if acq_options is not None else {}
        self.mc_samples = mc_samples
        self.rseed = rseed

    @abstractmethod
    def generate_candidates(self, model, train_x, train_obj): ...


class QLogEHVIAcquisition(AcquisitionFunction):
    def __init__(
        self,
        ref_point=None,
        bounds=None,
        batch_size=None,
        num_restarts=None,
        raw_samples=None,
        sequential=None,
        equality_constraints=None,
        acq_options=None,
        mc_samples=128,
        rseed=123,
    ):
        super().__init__(
            ref_point=ref_point,
            bounds=bounds,
            batch_size=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=sequential,
            equality_constraints=equality_constraints,
            acq_options=acq_options,
            mc_samples=mc_samples,
            rseed=rseed,
        )
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.mc_samples]),
            seed=self.rseed,
        )

    def generate_candidates(self, model, train_x, train_obj, time_dict):
        if model is None:
            raise ValueError("qLogEHVI acquisition requires a surrogate model.")
        t0 = time.time()
        with torch.no_grad():
            posterior_mean = model.posterior(train_x).mean
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
        options = {"maxiter": 200, **self.acq_options}
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=options,
            sequential=self.sequential,
            equality_constraints=self.equality_constraints,
        )
        if "acquisition_optimization" not in time_dict:
            time_dict["acquisition_optimization"] = time.time() - t0
        else:
            time_dict["acquisition_optimization"] += time.time() - t0

        return candidates


class RandomDirichletAcquisition(AcquisitionFunction):
    def __init__(
        self,
        ref_point=None,
        bounds=None,
        batch_size=None,
        num_restarts=None,
        raw_samples=None,
        sequential=None,
        equality_constraints=None,
        acq_options=None,
        mc_samples=128,
        rseed=123,
    ):
        super().__init__(
            ref_point=ref_point,
            bounds=bounds,
            batch_size=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=sequential,
            equality_constraints=equality_constraints,
            acq_options=acq_options,
            mc_samples=mc_samples,
            rseed=rseed,
        )
        self.simplex_dim = self.bounds.shape[-1]

    def generate_candidates(self, model, train_x, train_obj, time_dict):
        t0 = time.time()
        base = torch.ones(self.simplex_dim, dtype=train_x.dtype, device=train_x.device)
        distribution = torch.distributions.dirichlet.Dirichlet(base)
        samples = distribution.sample((self.batch_size,))

        if "acquisition_optimization" not in time_dict:
            time_dict["acquisition_optimization"] = time.time() - t0
        else:
            time_dict["acquisition_optimization"] += time.time() - t0
        return samples


ACQUISITION_REGISTRY = {
    "qlogehvi": QLogEHVIAcquisition,
    "random": RandomDirichletAcquisition,
}


def build_acquisition(name, config):
    key = name.lower()
    if key not in ACQUISITION_REGISTRY:
        raise ValueError(f"Unknown acquisition function '{name}'")
    return ACQUISITION_REGISTRY[key](config)


def available_acquisitions():
    return tuple(ACQUISITION_REGISTRY.keys())
