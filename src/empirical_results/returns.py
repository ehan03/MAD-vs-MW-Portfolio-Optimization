# standard library imports
from typing import Tuple

# third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# local imports
from src.models import MADModel, MarkowitzModel


class Returns:
    """Class to compare returns of MAD and Markowitz models."""

    def __init__(
        self,
        daily_prices: pd.DataFrame,
        split_date: str,
        min_return: float,
    ) -> None:
        """
        Initialize Returns class.

        Attributes:
            daily_prices (pd.DataFrame): Daily closing prices
            split_date (str): Date to split data into estimation and backtesting
            min_return (float): Minimum return to consider for portfolio weights
        """

        self.daily_prices = daily_prices
        self.split_date = pd.to_datetime(split_date)
        self.min_return = min_return

    def create_returns_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute returns from daily closing prices and split into
        data for estimation and backtesting.

        Returns:
            Tuple of dataframes with returns data for estimation and backtesting
        """

        returns = self.daily_prices.pct_change().dropna()
        estimation_data = returns.loc[: self.split_date]
        test_data = returns.loc[self.split_date :]

        return estimation_data, test_data

    def get_portfolio_weights(
        self, estimation_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute portfolio weights for MAD and Markowitz models given estimation data.

        Parameters:
            estimation_data (pd.DataFrame): Data for estimation
        """

        mad_model = MADModel(estimation_data, self.min_return)
        markowitz_model = MarkowitzModel(
            estimation_data, self.min_return, "reformulation"
        )

        mad_weights, _, _ = mad_model()
        markowitz_weights, _, _ = markowitz_model()

        return mad_weights, markowitz_weights

    def test_portfolios(
        self,
        test_data: pd.DataFrame,
        mad_weights: np.ndarray,
        markowitz_weights: np.ndarray,
    ) -> None:
        """
        Compare returns of MAD and Markowitz portfolios given test data

        Parameters:
            test_data (pd.DataFrame): Data for backtesting
            mad_weights (np.ndarray): MAD portfolio weights
            markowitz_weights (np.ndarray): Markowitz portfolio weights
        """

        mad_returns = (test_data * mad_weights).sum(axis=1)
        markowitz_returns = (test_data * markowitz_weights).sum(axis=1)

        mad_cumulative_returns = (mad_returns + 1).cumprod()
        markowitz_cumulative_returns = (markowitz_returns + 1).cumprod()

        plt.figure(figsize=(8, 6))
        plt.plot(mad_cumulative_returns, label="MAD")
        plt.plot(markowitz_cumulative_returns, label="Markowitz")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Portfolio value")
        plt.xticks(rotation=45)
        plt.axhline(y=1, color="black", linestyle="--")
        plt.show()

    def __call__(self) -> None:
        """
        Plot returns of MAD and Markowitz models given daily prices
        """

        estimation_data, test_data = self.create_returns_datasets()
        mad_weights, markowitz_weights = self.get_portfolio_weights(estimation_data)
        self.test_portfolios(test_data, mad_weights, markowitz_weights)
