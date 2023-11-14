# standard library imports

# third party imports
import numpy as np
import pandas as pd

# local imports


def simulate_returns(distribution: str, n_assets: int, n_obs: int) -> pd.DataFrame:
    """
    Simulate returns data from a given distribution

    Parameters:
        distribution (str): Distribution to simulate returns data from

    Returns:
        DataFrame containing the simulated returns data
    """

    assert distribution in ["normal", "lognormal"]

    if distribution == "normal":
        returns_vec = np.random.normal(0, 1, size=(n_assets, n_obs))
    else:
        returns_vec = np.random.lognormal(0, 1, size=(n_assets, n_obs)) - 1

    return pd.DataFrame(returns_vec.T)


def generate_random_weights(n_assets: int) -> np.ndarray:
    """
    Generate random weights for the portfolio

    Parameters:
        n_assets (int): Number of assets in the portfolio

    Returns:
        Array containing the random weights
    """

    weights = np.random.random(n_assets)
    weights /= np.sum(weights)

    return weights
