import pandas as pd
from typing import Callable

def backtest(data: pd.DataFrame, strategy: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    """
    Perform backtesting on a given strategy.
    
    :param data: DataFrame containing historical price data
    :param strategy: A function that takes a DataFrame and returns a Series of positions
    :return: DataFrame with backtesting results
    """
    positions = strategy(data)
    data['Position'] = positions
    data['Returns'] = data['Close'].pct_change()
    data['Strategy Returns'] = data['Position'].shift(1) * data['Returns']
    data['Cumulative Returns'] = (1 + data['Strategy Returns']).cumprod()
    return data

# You can add more backtesting-related functions here as needed