import pandas as pd
import requests
import logging
from fredapi import Fred

logger = logging.getLogger(__name__)

class EconomicIndicators:
    def __init__(self, fred_api_key):
        self.fred = Fred(api_key=fred_api_key)

    def get_indicator(self, indicator_id, start_date, end_date):
        try:
            data = self.fred.get_series(indicator_id, start_date, end_date)
            return data
        except Exception as e:
            logger.error(f"Error fetching indicator {indicator_id}: {str(e)}")
            return pd.Series()

    def get_inflation_rate(self, start_date, end_date):
        return self.get_indicator('CPIAUCSL', start_date, end_date).pct_change()

    def get_usd_index(self, start_date, end_date):
        return self.get_indicator('DTWEXBGS', start_date, end_date)

    def get_interest_rate(self, start_date, end_date):
        return self.get_indicator('FEDFUNDS', start_date, end_date)

    def get_gdp_growth(self, start_date, end_date):
        return self.get_indicator('GDP', start_date, end_date).pct_change()

    def get_unemployment_rate(self, start_date, end_date):
        return self.get_indicator('UNRATE', start_date, end_date)

    def get_all_indicators(self, start_date, end_date):
        indicators = pd.DataFrame({
            'inflation_rate': self.get_inflation_rate(start_date, end_date),
            'usd_index': self.get_usd_index(start_date, end_date),
            'interest_rate': self.get_interest_rate(start_date, end_date),
            'gdp_growth': self.get_gdp_growth(start_date, end_date),
            'unemployment_rate': self.get_unemployment_rate(start_date, end_date)
        })
        return indicators.fillna(method='ffill').fillna(method='bfill')

# You can add more economic indicator functions here as needed