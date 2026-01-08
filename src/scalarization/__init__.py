def build_scalarizer(cfg, instance, env=None, maximization=None):
    if cfg.problem.name == "mokp":
        maximization = True if maximization is None else maximization
        if cfg.scalarization.name == "aug_cheby":
            if cfg.optimizer == "gurobi":
                from .aug_cheby.mokp import GurobiAugChebyMOKPScalarizer

                return GurobiAugChebyMOKPScalarizer(
                    instance, env, rho=cfg.scalarization.rho, maximization=maximization
                )
            elif cfg.optimizer == "scip":
                from .aug_cheby.mokp import SCIPAugChebyMOKPScalarizer

                return SCIPAugChebyMOKPScalarizer(
                    instance,
                    rho=cfg.scalarization.rho,
                    time_limit=cfg.time_limit,
                    maximization=maximization,
                )
            else:
                raise ValueError("Invalid solver")
        else:
            raise ValueError("Invalid scalarizer")
    if cfg.problem.name == "moap":
        maximization = False if maximization is None else maximization
        if cfg.scalarization.name == "aug_cheby":
            if cfg.optimizer == "gurobi":
                from .aug_cheby.moap import GurobiAugChebyMOAPScalarizer

                return GurobiAugChebyMOAPScalarizer(
                    instance, env, rho=cfg.scalarization.rho, maximization=maximization
                )
            elif cfg.optimizer == "scip":
                from .aug_cheby.moap import SCIPAugChebyMOAPScalarizer

                return SCIPAugChebyMOAPScalarizer(
                    instance,
                    rho=cfg.scalarization.rho,
                    time_limit=cfg.time_limit,
                    maximization=maximization,
                )
            else:
                raise ValueError("Invalid solver")
        else:
            raise ValueError("Invalid scalarizer")
    else:
        raise ValueError("Invalid problem!")
