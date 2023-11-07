"""Mean absolute deviation model"""

# standard library imports

# third party imports
import cvxpy as cp
import numpy as np

# local imports
from src.models.base import BaseModel


class MADModel(BaseModel):
    """
    Mean absolute deviation model
    """

    def find_optimal_weights(self, mean_returns: np.ndarray) -> np.ndarray:
        """
        Find optimal weights

        Parameters:
            mean_returns (np.ndarray): Mean returns

        Returns:
            NumPy array of optimal weights
        """

        T, n = self.returns_data.loc[: self.split_date].shape
        a = (self.returns_data.loc[: self.split_date] - mean_returns).values
        w = cp.Variable(n)
        y = cp.Variable(T)

        constraints = [
            y + a @ w >= 0,
            y - a @ w >= 0,
            w @ mean_returns >= self.min_return,
            cp.sum(w) == 1,
            w >= 0,
        ]

        objective = cp.Minimize(cp.sum(y) / T)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return w.value

    def __call__(self) -> np.ndarray:
        """
        Find optimal weights

        Returns:
            NumPy array of optimal weights
        """

        mean_returns = self.estimate_mean_returns()
        weights = self.find_optimal_weights(mean_returns)

        return weights
