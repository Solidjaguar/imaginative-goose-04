import pandas as pd
import numpy as np
from typing import Callable

def backtest(data: pd.DataFrame, strategy: Callable[[pd.DataFrame], pd.Series], initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Perform backtesting on a given strategy.
    
    :param data: DataFrame containing historical price data
    :param strategy: A function that takes a DataFrame and returns a Series of positions
    :param initial_capital: Initial capital for the backtest
    :return: DataFrame with backtesting results
    """
    positions = strategy(data)
    data['Position'] = positions
    data['Returns'] = data['Close'].pct_change()
    data['Strategy Returns'] = data['Position'].shift(1) * data['Returns']
    data['Cumulative Returns'] = (1 + data['Strategy Returns']).cumprod()
    data['Equity'] = initial_capital * data['Cumulative Returns']
    
    # Calculate additional metrics
    data['Drawdown'] = (data['Equity'].cummax() - data['Equity']) / data['Equity'].cummax()
    
    return data

def calculate_performance_metrics(backtest_results: pd.DataFrame) -> dict:
    """
    Calculate performance metrics from backtest results.
    
    :param backtest_results: DataFrame with backtest results
    :return: Dictionary of performance metrics
    """
    total_return = backtest_results['Cumulative Returns'].iloc[-1] - 1
    sharpe_ratio = np.sqrt(252) * backtest_results['Strategy Returns'].mean() / backtest_results['Strategy Returns'].std()
    max_drawdown = backtest_results['Drawdown'].max()
    
    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

# You can add more backtesting-related functions here as needed