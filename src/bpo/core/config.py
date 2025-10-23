import torch


class BOConfig:
    def __init__(
        self,
        acquisition="qlogehvi",
        rseed=123,
        n_initial_samples=10,
        n_iterations=20,
        mc_samples=128,
        batch_size_q=2,
        num_restarts=10,
        raw_samples=512,
        should_maximize=True,
        sequential=True,
        ref_point=None,
    ):
        self.acquisition = acquisition
        self.rseed = rseed
        self.n_initial_samples = n_initial_samples
        self.n_iterations = n_iterations
        self.mc_samples = mc_samples
        self.batch_size_q = batch_size_q
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.should_maximize = should_maximize
        self.sequential = sequential
        self.ref_point = None
        if ref_point is not None:
            self.ref_point = ref_point.to(dtype=torch.get_default_dtype())
