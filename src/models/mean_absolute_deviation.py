# standard library imports
from typing import Tuple

# third party imports
import cvxpy as cp
import numpy as np
import pandas as pd

# local imports
from src.models.base import BaseModel


class MADModel(BaseModel):
    """Class for the mean absolute deviation model"""

    def solve(self, mean_returns: pd.Series) -> Tuple[np.ndarray, float, float]:
        """
        Solve MAD model

        Parameters:
            mean_returns (pd.Series): Mean returns

        Returns:
            Tuple of optimal weights, expected return, and risk (L1)
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

        return w.value, w.value @ mean_returns, problem.value

    def __call__(self) -> Tuple[np.ndarray, float, float]:
        """
        Find optimal weights

        Returns:
            Tuple of optimal weights, expected return, and risk (L1)
        """

        mean_returns = self.estimate_mean_returns()
        weights, expected_return, risk = self.solve(mean_returns)

        return weights, expected_return, risk
