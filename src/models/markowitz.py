"""Markowitz model"""

# standard library imports

# third party imports
import cvxpy as cp
import numpy as np
import pandas as pd

# local imports
from src.models.base import BaseModel


class MarkowitzModel(BaseModel):
    """Markowitz model"""

    def __init__(
        self,
        returns_data: pd.DataFrame,
        min_return: float,
        covariance_method: str,
    ) -> None:
        """
        Initialize class

        Attributes:
            returns_data (pd.DataFrame): Returns data
            min_return (float): Minimum return for optimization formulation
            use_covariance (bool): Whether to use covariance matrix or not
        """

        super().__init__(returns_data, min_return)
        assert covariance_method in ["regular", "perturbed", "reformulation"]
        self.covariance_method = covariance_method

    def estimate_cov_matrix(self) -> pd.DataFrame:
        """
        Estimate covariance matrix

        Returns:
            NumPy array of covariance matrix
        """

        cov_matrix = self.returns_data.cov()

        return cov_matrix

    def solve_regular(self, mean_returns: pd.Series) -> np.ndarray:
        """
        Solve Markowitz model with regular covariance matrix

        Parameters:
            mean_returns (pd.Series): Mean returns

        Returns:
            NumPy array of optimal weights
        """

        cov_matrix = self.estimate_cov_matrix()
        w = cp.Variable(self.N)

        constraints = [
            w @ mean_returns >= self.min_return,
            cp.sum(w) == 1,
            w >= 0,
        ]

        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return w.value

    def solve_perturbed(
        self, mean_returns: pd.Series, epsilon: float = 1e-8
    ) -> np.ndarray:
        """
        Solve Markowitz model with perturbed covariance matrix

        Parameters:
            mean_returns (pd.Series): Mean returns
            epsilon (float): Perturbation parameter, how much to shift the diagonal

        Returns:
            NumPy array of optimal weights
        """

        assert epsilon > 0
        cov_matrix = self.estimate_cov_matrix()
        w = cp.Variable(self.N)
        sigma = cov_matrix + epsilon * np.eye(self.N)

        constraints = [
            w @ mean_returns >= self.min_return,
            cp.sum(w) == 1,
            w >= 0,
        ]

        objective = cp.Minimize(cp.quad_form(w, sigma))
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return w.value

    def solve_reformulation(self, mean_returns: pd.Series) -> np.ndarray:
        """
        Solve Markowitz model with reformulation to avoid numerical issues

        Parameters:
            mean_returns (pd.Series): Mean returns

        Returns:
            NumPy array of optimal weights
        """

        a = (self.returns_data - mean_returns).values
        w = cp.Variable(self.N)
        m = cp.Variable(self.T)

        constraints = [
            m == a @ w,
            w @ mean_returns >= self.min_return,
            cp.sum(w) == 1,
            w >= 0,
        ]

        objective = cp.Minimize(cp.sum(m**2) / (self.T - 1))
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

        if self.covariance_method == "regular":
            weights = self.solve_regular(mean_returns)
        elif self.covariance_method == "perturbed":
            weights = self.solve_perturbed(mean_returns)
        elif self.covariance_method == "reformulation":
            weights = self.solve_reformulation(mean_returns)

        return weights
