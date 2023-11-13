"""Base portfolio optimization model class"""

# standard library imports

# third party imports
import numpy as np
import pandas as pd

# local imports


class BaseModel:
    """
    Base portfolio optimization model class

    Attributes:
        returns_data (pd.DataFrame): Returns data
        min_return (float): Minimum return for optimization formulation
    """

    def __init__(self, returns_data: pd.DataFrame, min_return: float) -> None:
        """
        Initialize BaseModel class

        Parameters:
            returns_data (pd.DataFrame): Returns data
            min_return (float): Minimum return for optimization formulation
        """

        self.returns_data = returns_data
        self.min_return = min_return
        self.T, self.N = returns_data.shape

    def estimate_mean_returns(self) -> pd.Series:
        """
        Estimate mean returns

        Returns:
            NumPy array of mean returns up to split date
        """

        mean_returns = self.returns_data.mean()

        return mean_returns
