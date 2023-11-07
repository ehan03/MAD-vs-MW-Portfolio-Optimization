"""Base class for creating dataset from Yahoo Finance API."""

# standard imports
import os
from typing import List

# third-party imports
import pandas as pd
import yfinance as yf

# local imports


class DataGenerator:
    """
    Base class for creating dataset for NASDAQ stocks using the Yahoo Finance API.
    The tickers were downloaded from https://github.com/rreichel3/US-Stock-Symbols,
    so it is up to date as of 10/17/2023.
    """

    def __init__(self) -> None:
        """
        Initialize the DataGenerator class.
        """

        self.input_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "nasdaq_screener.csv",
        )
        self.output_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "nasdaq_adj_closing_prices.csv",
        )

    def get_tickers(self, file_path: str) -> List[str]:
        """
        Get list of tickers from CSV file downloaded from NASDAQ screener
        website (https://www.nasdaq.com/market-activity/stocks/screener) for
        NASDAQ stocks in the United States with a market cap of at least $300M
        (Small to Mega Cap).

        Parameters:
            file_path (str): Path to CSV file downloaded from NASDAQ website

        Returns:
            List of string tickers
        """

        df = pd.read_csv(file_path, usecols=["Symbol"])
        tickers = df["Symbol"].tolist()

        return tickers

    def get_historical_price_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Gets historical price data for each ticker in the list and
        returns a dataframe

        Parameters:
            tickers (List[str]): List of tickers

        Returns:
            Pandas dataframe containing historical monthly price data
        """

        price_df = yf.download(
            tickers=tickers,
            period="max",
            interval="1mo",
        )

        return price_df

    def __call__(self) -> None:
        """
        Generate NASDAQ stock dataset
        """

        tickers = self.get_tickers(self.input_path)
        price_df = self.get_historical_price_data(tickers)
        price_df["Adj Close"].to_csv(self.output_path)
