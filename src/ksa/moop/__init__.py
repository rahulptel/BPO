from .knapsack import Knapsack


def factory(problem, filename):
    if problem == 'KP':
        return Knapsack(filename)
    else:
        raise ValueError('Invalid problem name!')
