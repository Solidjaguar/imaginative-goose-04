import pandas as pd
import numpy as np
import requests
from io import StringIO
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import logging

logger = logging.getLogger(__name__)

class AdvancedDataFetcher:
    def __init__(self, alpha_vantage_api_key):
        self.alpha_vantage_api_key = alpha_vantage_api_key
        self.ts = TimeSeries(key=self.alpha_vantage_api_key, output_format='pandas')

    def fetch_gold_data(self, start_date, end_date):
        gold = yf.Ticker("GC=F")
        gold_data = gold.history(start=start_date, end=end_date)
        return gold_data

    def fetch_currency_data(self, currency_pair, start_date, end_date):
        currency = yf.Ticker(f"{currency_pair}=X")
        currency_data = currency.history(start=start_date, end=end_date)
        return currency_data

    def fetch_stock_index_data(self, index_symbol, start_date, end_date):
        index = yf.Ticker(index_symbol)
        index_data = index.history(start=start_date, end=end_date)
        return index_data

    def fetch_economic_data(self, indicator):
        try:
            data, _ = self.ts.get_daily(symbol=indicator, outputsize='full')
            return data
        except Exception as e:
            logger.error(f"Error fetching economic data: {str(e)}")
            return pd.DataFrame()

    def fetch_commitment_of_traders(self):
        url = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
        response = requests.get(url)
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text), sep=",", header=0)
            gold_data = data[data['Market_and_Exchange_Names'] == 'GOLD - COMMODITY EXCHANGE INC.']
            return gold_data
        else:
            logger.error("Failed to fetch Commitment of Traders data")
            return pd.DataFrame()

    def fetch_gold_etf_flows(self, etf_symbol, start_date, end_date):
        etf = yf.Ticker(etf_symbol)
        etf_data = etf.history(start=start_date, end=end_date)
        return etf_data

    def fetch_all_data(self, start_date, end_date):
        gold_data = self.fetch_gold_data(start_date, end_date)
        usd_eur = self.fetch_currency_data("USDEUR", start_date, end_date)
        sp500 = self.fetch_stock_index_data("^GSPC", start_date, end_date)
        gold_etf = self.fetch_gold_etf_flows("GLD", start_date, end_date)
        cot_data = self.fetch_commitment_of_traders()
        
        # Fetch some economic indicators
        gdp = self.fetch_economic_data('GDP')
        inflation = self.fetch_economic_data('CPIAUCSL')
        interest_rate = self.fetch_economic_data('FEDFUNDS')

        # Combine all data
        combined_data = pd.concat([
            gold_data['Close'],
            usd_eur['Close'].rename('USDEUR'),
            sp500['Close'].rename('SP500'),
            gold_etf['Volume'].rename('GLD_Volume'),
            gdp.rename(columns={'1. open': 'GDP'}),
            inflation.rename(columns={'1. open': 'Inflation'}),
            interest_rate.rename(columns={'1. open': 'InterestRate'})
        ], axis=1)

        # Merge COT data (this might need more processing depending on the format)
        combined_data = pd.merge(combined_data, cot_data, left_index=True, right_on='Report_Date_as_YYYY-MM-DD', how='left')

        return combined_data.fillna(method='ffill')

# You can add more data fetching methods here as needed