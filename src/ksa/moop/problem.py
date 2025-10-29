from abc import ABC, abstractmethod


class Problem(ABC):
    @abstractmethod
    def _load_data(self):
        """Load problem data"""
        pass

    @abstractmethod
    def get_bounds(self):
        """Get the lower and upper bound for each objective function"""
        pass

    @abstractmethod
    def get_solution(self, epsilon=None):
        """Get a (efficient, nondominated) solution tuple for a
        given epsilon"""
        pass
