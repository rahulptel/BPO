from .mokp import MOKP

PROBLEM_REGISTRY = {
    "mokp": MOKP,
}


def available_problems():
    return tuple(PROBLEM_REGISTRY.keys())


def build_problem(name, **kwargs):
    key = name.lower()
    if key not in PROBLEM_REGISTRY:
        raise ValueError(f"Unknown problem '{name}'")
    return PROBLEM_REGISTRY[key](**kwargs)
