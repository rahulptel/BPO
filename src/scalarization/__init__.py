def build_scalarizer(cfg, instance, env=None):
    if cfg.optimizer != "gurobi":
        raise ValueError("Only the Gurobi optimizer is supported.")

    if cfg.problem.name == "mokp":
        if cfg.scalarization.name == "aug_cheby":
            from .aug_cheby.mokp import GurobiAugChebyMOKPScalarizer

            return GurobiAugChebyMOKPScalarizer(
                instance, env, rho=cfg.scalarization.rho
            )
        else:
            raise ValueError("Invalid scalarizer")
    if cfg.problem.name == "moap":
        if cfg.scalarization.name == "aug_cheby":
            from .aug_cheby.moap import GurobiAugChebyMOAPScalarizer

            return GurobiAugChebyMOAPScalarizer(
                instance, env, rho=cfg.scalarization.rho
            )
        else:
            raise ValueError("Invalid scalarizer")
    else:
        raise ValueError("Invalid problem!")
