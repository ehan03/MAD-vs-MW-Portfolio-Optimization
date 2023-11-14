# standard library imports

# third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# local imports
from src.models import MADModel, MarkowitzModel
from src.utils import generate_random_weights, simulate_returns


class EfficientFrontier:
    """Class for plotting the efficient frontier of the portfolio"""

    def __init__(self) -> None:
        """
        Initialize the EfficientFrontier class
        """

        self.n_portfolios = 5000

    def draw_efficient_frontier(
        self,
        returns_data: pd.DataFrame,
        model: str,
    ) -> None:
        """
        Plot the efficient frontier of the portfolio

        Parameters:
            returns_data (DataFrame): DataFrame containing the returns data
            min_returns (List[float]): List of minimum returns to plot
            model (str): Model to use for the efficient frontier
        """

        assert model in ["markowitz", "mad"]

        mean_returns = returns_data.mean()
        cov_matrix = returns_data.cov()

        risk_label = (
            "Standard Deviation" if model == "markowitz" else "Mean Absolute Deviation"
        )

        # Generate random portfolios
        random_expected_returns = []
        random_risks = []
        for _ in range(self.n_portfolios):
            random_weights = generate_random_weights(len(mean_returns))
            random_expected_returns.append(random_weights @ mean_returns)

            if model == "markowitz":
                random_risks.append(
                    np.sqrt(random_weights.T @ cov_matrix @ random_weights)
                )
            else:
                random_risks.append(
                    np.mean(np.abs((returns_data - mean_returns) @ random_weights))
                )

        ef_x = []
        ef_y = []
        min_returns = np.linspace(
            max(0, np.mean(random_expected_returns)),
            np.max(random_expected_returns),
            20,
        )
        for min_return in min_returns:
            if model == "markowitz":
                mw_model = MarkowitzModel(returns_data, min_return, "regular")
                _, expected_return, risk = mw_model()
                ef_x.append(expected_return)
                ef_y.append(risk)
            else:
                mad_model = MADModel(returns_data, min_return)
                _, expected_return, risk = mad_model()
                ef_x.append(expected_return)
                ef_y.append(risk)

        plt.scatter(random_expected_returns, random_risks, marker=".", color="blue")
        plt.plot(
            ef_x,
            ef_y,
            marker="o",
            color="red",
            linestyle="dashed",
            label="Efficient Frontier",
        )
        plt.xlabel("Expected Return (%)")
        plt.ylabel(risk_label)
        plt.legend(loc="upper left")
        plt.show()
