"""Mean absolute deviation model"""

# standard library imports

# third party imports
import cvxpy as cp
import numpy as np
import pandas as pd

# local imports
from src.models.base import BaseModel


class MADModel(BaseModel):
    """Mean absolute deviation model"""

    def solve(self, mean_returns: pd.Series) -> np.ndarray:
        """
        Solve MAD model

        Parameters:
            mean_returns (pd.Series): Mean returns

        Returns:
            NumPy array of optimal weights
        """

        a = (self.returns_data - mean_returns).values
        w = cp.Variable(self.N)
        y = cp.Variable(self.T)

        constraints = [
            y + a @ w >= 0,
            y - a @ w >= 0,
            w @ mean_returns >= self.min_return,
            cp.sum(w) == 1,
            w >= 0,
        ]

        objective = cp.Minimize(cp.sum(y) / self.T)
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
        weights = self.solve(mean_returns)

        return weights
