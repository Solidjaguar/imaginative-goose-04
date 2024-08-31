import pandas as pd
import yfinance as yf

def fetch_gold_data(start_date, end_date):
    """
    Fetch gold price data from Yahoo Finance.
    
    :param start_date: Start date for data fetching (YYYY-MM-DD)
    :param end_date: End date for data fetching (YYYY-MM-DD)
    :return: DataFrame with gold price data
    """
    gold_data = yf.download("GC=F", start=start_date, end=end_date)
    return gold_data

def preprocess_data(data):
    """
    Preprocess the gold price data.
    
    :param data: DataFrame with raw gold price data
    :return: DataFrame with preprocessed data
    """
    # Add technical indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    
    # Remove rows with NaN values
    data.dropna(inplace=True)
    
    return data

def calculate_rsi(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.
    
    :param prices: Series of prices
    :param period: RSI period (default: 14)
    :return: Series with RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# You can add more data handling functions here as needed