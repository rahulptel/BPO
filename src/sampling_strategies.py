from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)


@dataclass
class StrategyConfig:
    ref_point: torch.Tensor
    bounds: torch.Tensor
    batch_size: int
    num_restarts: int
    raw_samples: int
    sequential: bool
    equality_constraints: Optional[Sequence] = None
    acq_options: Dict[str, int] = field(default_factory=dict)
    mc_samples: int = 128
    seed: Optional[int] = None


class SamplingStrategy(ABC):
    def __init__(self, config: StrategyConfig):
        self.config = config

    @abstractmethod
    def generate_candidates(
        self, model, train_x: torch.Tensor, train_obj: torch.Tensor
    ) -> torch.Tensor:
        ...


class QLogEHVIStrategy(SamplingStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.config.mc_samples]),
            seed=self.config.seed,
        )

    def generate_candidates(
        self, model, train_x: torch.Tensor, train_obj: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            posterior_mean = model.posterior(train_x).mean
        partitioning = FastNondominatedPartitioning(
            ref_point=self.config.ref_point,
            Y=posterior_mean,
        )
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.config.ref_point,
            partitioning=partitioning,
            sampler=self.sampler,
        )
        options = {"maxiter": 200, **self.config.acq_options}
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.config.bounds,
            q=self.config.batch_size,
            num_restarts=self.config.num_restarts,
            raw_samples=self.config.raw_samples,
            options=options,
            sequential=self.config.sequential,
            equality_constraints=self.config.equality_constraints,
        )
        return candidates


class RandomDirichletStrategy(SamplingStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.simplex_dim = self.config.bounds.shape[-1]

    def generate_candidates(
        self, model, train_x: torch.Tensor, train_obj: torch.Tensor
    ) -> torch.Tensor:
        base = torch.ones(self.simplex_dim, dtype=train_x.dtype, device=train_x.device)
        distribution = torch.distributions.dirichlet.Dirichlet(base)
        samples = distribution.sample((self.config.batch_size,))
        return samples


STRATEGY_REGISTRY = {
    "qlogehvi": QLogEHVIStrategy,
    "random": RandomDirichletStrategy,
}


def build_strategy(name: str, config: StrategyConfig) -> SamplingStrategy:
    key = name.lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown sampling strategy '{name}'")
    return STRATEGY_REGISTRY[key](config)


def available_strategies() -> Sequence[str]:
    return tuple(STRATEGY_REGISTRY.keys())
