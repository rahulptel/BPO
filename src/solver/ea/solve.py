from ea.core.run import run_ea

from .adapters import PymooMOKPProblem


class EASolver:
    def __init__(self, cfg, instance):
        self.cfg = cfg
        self.instance = instance

    def run(self):
        problem = PymooMOKPProblem(self.instance)
        run_ea(problem, self.cfg)
