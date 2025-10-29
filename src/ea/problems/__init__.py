from .mokp import MOKP


PROBLEM_REGISTRY = {
    "mokp": MOKP,
}


def available_problems():
    return tuple(PROBLEM_REGISTRY.keys())


def build_problem(name, **kwargs):
    key = str(name).lower()
    if key not in PROBLEM_REGISTRY:
        available = ", ".join(sorted(PROBLEM_REGISTRY))
        raise ValueError(f"Unknown EA problem '{name}'. Available: {available}")
    return PROBLEM_REGISTRY[key](**kwargs)
