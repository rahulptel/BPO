from pathlib import Path

import torch


class Problem:
    name = "base"

    def n_objectives(self):
        raise NotImplementedError("Problem subclasses must define n_objectives.")

    def lambda_bounds(self):
        raise NotImplementedError("Problem subclasses must define lambda_bounds.")

    def lambda_equality_constraints(self):
        return None

    def default_ref_point(self):
        return torch.zeros(self.n_objectives(), dtype=torch.get_default_dtype())

    def ideal_point(self):
        return None

    def initial_design(self, n):
        raise NotImplementedError("Problem subclasses must define initial_design.")

    def evaluate(self, lambda_batch, maximize=True):
        raise NotImplementedError("Problem subclasses must define evaluate.")

    def metadata(self):
        return {}

    def io_base_dir(self, config):
        return Path("../outputs") / self.name
