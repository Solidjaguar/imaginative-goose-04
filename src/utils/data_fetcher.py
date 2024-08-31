import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, symbol="GC=F", start_date=None, end_date=None):
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')

    def fetch_data(self):
        try:
            logger.info(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}")
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            
            if data.empty:
                raise ValueError("No data retrieved. Please check your inputs.")
            
            logger.info(f"Successfully fetched {len(data)} rows of data")
            return data
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    def check_data_quality(self, data):
        logger.info("Checking data quality...")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Missing values detected:\n{missing_values}")
        
        # Check for duplicate dates
        duplicate_dates = data.index.duplicated()
        if duplicate_dates.sum() > 0:
            logger.warning(f"Duplicate dates detected: {duplicate_dates.sum()} duplicates")
        
        # Check for outliers (example: prices outside 3 standard deviations)
        mean = data['Close'].mean()
        std = data['Close'].std()
        outliers = data[(data['Close'] < mean - 3*std) | (data['Close'] > mean + 3*std)]
        if not outliers.empty:
            logger.warning(f"Potential outliers detected: {len(outliers)} data points")
        
        logger.info("Data quality check completed")

    def get_data(self):
        data = self.fetch_data()
        self.check_data_quality(data)
        return data

# You can add more data fetching functions here as needed