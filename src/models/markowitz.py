"""Markowitz model"""

# standard library imports

# third party imports
import cvxpy as cp
import numpy as np
import pandas as pd

# local imports
from src.models.base import BaseModel


class MarkowitzModel(BaseModel):
    """
    Markowitz model
    """

    def estimate_cov_matrix(self) -> pd.DataFrame:
        """
        Estimate covariance matrix

        Returns:
            NumPy array of covariance matrix
        """

        cov_matrix = self.returns_data.loc[: self.split_date].cov()

        return cov_matrix

    def find_optimal_weights(
        self, mean_returns: pd.Series, cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """
        Find optimal weights

        Parameters:
            cov_matrix (np.ndarray): Covariance matrix

        Returns:
            NumPy array of optimal weights
        """

        n = self.returns_data.shape[1]
        w = cp.Variable(n)

        constraints = [
            w @ mean_returns >= self.min_return,
            cp.sum(w) == 1,
            w >= 0,
        ]

        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return w.value

    def __call__(self) -> np.ndarray:
        """
        Find optimal weights via Markowitz model

        Returns:
            NumPy array of optimal weights
        """

        mean_returns = self.estimate_mean_returns()
        cov_matrix = self.estimate_cov_matrix()
        weights = self.find_optimal_weights(mean_returns, cov_matrix)

        return weights
