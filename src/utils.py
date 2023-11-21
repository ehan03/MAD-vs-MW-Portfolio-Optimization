# standard library imports

# third party imports
import numpy as np
import pandas as pd

# local imports


def simulate_returns(
    distribution: str, n_assets: int, n_obs: int, random_cov: bool = False
) -> pd.DataFrame:
    """
    Simulate returns data from a given distribution

    Parameters:
        distribution (str): Distribution to simulate returns data from

    Returns:
        DataFrame containing the simulated returns data
    """

    assert distribution in ["normal", "lognormal"]

    np.random.seed(431)

    mean_returns = np.zeros(n_assets)

    if not random_cov:
        cov_matrix = np.eye(n_assets)
    else:
        A = np.random.random((n_assets, n_assets))
        cov_matrix = A.T @ A

    # Generate returns data
    if distribution == "normal":
        returns_vec = np.random.multivariate_normal(mean_returns, cov_matrix, n_obs)
    else:
        returns_vec = np.random.multivariate_normal(mean_returns, cov_matrix, n_obs)
        returns_vec = np.exp(returns_vec) - 1

    return pd.DataFrame(returns_vec)


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
