import numpy as np

def generate_instance(n_items, n_objectives, random_seed):
    """
    Generates a random knapsack problem instance.
    """
    rng = np.random.RandomState(random_seed)
    profits = rng.randint(1, 101, size=(n_items, n_objectives))
    weights = rng.randint(1, 101, size=n_items)
    capacity = np.ceil(weights.sum() / 2)
    return profits, weights, capacity
