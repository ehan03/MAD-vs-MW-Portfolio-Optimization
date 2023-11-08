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

    def __init__(
        self,
        returns_data: pd.DataFrame,
        min_return: float,
        split_date: str,
        use_covariance: bool = True,
    ) -> None:
        """
        Initialize class

        Attributes:
            returns_data (pd.DataFrame): Returns data
            min_return (float): Minimum return for optimization formulation
            split_date (str): Split date from which to estimate mean returns
            use_covariance (bool): Whether to use covariance matrix or not
        """

        super().__init__(returns_data, min_return, split_date)
        self.use_covariance = use_covariance

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
            mean_returns (pd.Series): Mean returns
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

    def find_optimal_weights_reformulation(self, mean_returns: pd.Series) -> np.ndarray:
        """
        Find optimal weights with a reformulation to avoid numerical issues
        related to the covariance matrix

        Parameters:
            mean_returns (pd.Series): Mean returns

        Returns:
            NumPy array of optimal weights
        """

        T, n = self.returns_data.loc[: self.split_date].shape
        a = (self.returns_data.loc[: self.split_date] - mean_returns).values
        w = cp.Variable(n)
        aux = cp.Variable(T)

        constraints = [
            aux == a @ w,
            w @ mean_returns >= self.min_return,
            cp.sum(w) == 1,
            w >= 0,
        ]

        objective = cp.Minimize(cp.sum(aux**2))
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

        if self.use_covariance:
            cov_matrix = self.estimate_cov_matrix() + 1e-6 * np.eye(len(mean_returns))
            min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
            print(f"Minimum eigenvalue: {min_eig}")
            weights = self.find_optimal_weights(mean_returns, cov_matrix)
        else:
            weights = self.find_optimal_weights_reformulation(mean_returns)

        return weights
