from ea.core.run import run_ea

from .adapters import PymooMOKPProblem


def solve(instance, cfg):
    problem = PymooMOKPProblem(instance)
    run_ea(problem, cfg)
