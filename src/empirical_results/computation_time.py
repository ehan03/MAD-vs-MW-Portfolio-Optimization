# standard library imports
import time

# third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# local imports
from src.models import MADModel, MarkowitzModel
from src.utils import simulate_returns


class ComputationTime:
    """
    Class for comparing the computation time required for
    Markowitz and MAD models
    """

    def test_num_assets(self) -> None:
        """
        Compare computation time for different numbers of assets
        """

        N = 501
        T = 150

        # Generate returns data
        returns_data = simulate_returns("normal", N, T, random_cov=True)

        # Computation times
        num_assets = np.linspace(25, 500, 20, dtype=int)

        mean_markowitz_times = []
        sd_markowitz_times = []
        mean_mad_times = []
        sd_mad_times = []

        for n in num_assets:
            temp_mw = []
            temp_mad = []

            for i in range(5):
                temp_assets = returns_data.sample(n=n, axis=1)
                markowitz = MarkowitzModel(temp_assets, -10, "perturbed")
                mad = MADModel(temp_assets, -10)

                start_mw = time.process_time()
                _, _, _ = markowitz()
                end_mw = time.process_time()

                start_mad = time.process_time()
                _, _, _ = mad()
                end_mad = time.process_time()

                temp_mw.append(end_mw - start_mw)
                temp_mad.append(end_mad - start_mad)

            mean_markowitz_times.append(np.mean(temp_mw))
            sd_markowitz_times.append(np.std(temp_mw))
            mean_mad_times.append(np.mean(temp_mad))
            sd_mad_times.append(np.std(temp_mad))

        # Plot computation times
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            num_assets,
            mean_markowitz_times,
            yerr=sd_markowitz_times,
            capsize=4,
            marker="s",
            color="red",
            markersize=4,
            linewidth=1,
            linestyle="solid",
            label="Markowitz",
        )
        plt.errorbar(
            num_assets,
            mean_mad_times,
            yerr=sd_mad_times,
            capsize=4,
            marker="o",
            color="green",
            markersize=4,
            linewidth=1,
            linestyle="solid",
            label="MAD",
        )
        plt.xlabel("Number of assets, N")
        plt.ylabel("Computation time (s)")
        plt.legend()
        plt.show()

    def test_num_obs(self) -> None:
        """
        Compare computation time for different numbers of observations
        """

        N = 50
        T = 1001

        # Generate returns data
        returns_data = simulate_returns("normal", N, T, random_cov=True)

        # Computation times
        num_obs = np.linspace(50, 1000, 20, dtype=int)

        mean_markowitz_times = []
        sd_markowitz_times = []
        mean_mad_times = []
        sd_mad_times = []

        for t in num_obs:
            temp_mw = []
            temp_mad = []

            for i in range(5):
                temp_obs = returns_data.sample(n=t, axis=0)
                markowitz = MarkowitzModel(temp_obs, -10, "regular")
                mad = MADModel(temp_obs, -10)

                start_mw = time.process_time()
                _, _, _ = markowitz()
                end_mw = time.process_time()

                start_mad = time.process_time()
                _, _, _ = mad()
                end_mad = time.process_time()

                temp_mw.append(end_mw - start_mw)
                temp_mad.append(end_mad - start_mad)

            mean_markowitz_times.append(np.mean(temp_mw))
            sd_markowitz_times.append(np.std(temp_mw))
            mean_mad_times.append(np.mean(temp_mad))
            sd_mad_times.append(np.std(temp_mad))

        # Plot computation times
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            num_obs,
            mean_markowitz_times,
            yerr=sd_markowitz_times,
            capsize=4,
            marker="s",
            color="red",
            markersize=4,
            linewidth=1,
            linestyle="solid",
            label="Markowitz",
        )
        plt.errorbar(
            num_obs,
            mean_mad_times,
            yerr=sd_mad_times,
            capsize=4,
            marker="o",
            color="green",
            markersize=4,
            linewidth=1,
            linestyle="solid",
            label="MAD",
        )
        plt.xlabel("Number of observations, T")
        plt.ylabel("Computation time (s)")
        plt.legend()
        plt.show()
