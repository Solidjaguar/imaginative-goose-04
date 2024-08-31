import pandas as pd
import numpy as np
from src.utils.advanced_data_fetcher import AdvancedDataFetcher
from src.utils.advanced_feature_engineering import AdvancedFeatureEngineer
from src.utils.drift_detector import DriftDetector
from src.utils.sentiment_analyzer import SentimentAnalyzer
from src.utils.advanced_news_analyzer import AdvancedNewsAnalyzer
from src.models.ensemble_predictor import EnsemblePredictor
from src.models.auto_model_selector import AutoModelSelector
from src.models.pattern_detector import PatternDetector
from src.strategies.rl_trading_strategy import RLTradingStrategy
from src.backtesting.advanced_backtester import AdvancedBacktester
from src.forward_testing.forward_tester import ForwardTester
from src.risk_management.advanced_risk_manager import AdvancedRiskManager
from src.optimization.hyperparameter_optimizer import HyperparameterOptimizer
from src.utils.model_versioner import ModelVersioner
from src.utils.trading212_client import Trading212Client
import logging
import schedule
import time
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedGoldTradingSystem:
    def __init__(self, config):
        self.config = config
        self.data_fetcher = AdvancedDataFetcher(config['alpha_vantage_api_key'])
        self.feature_engineer = AdvancedFeatureEngineer()
        self.drift_detector = DriftDetector()
        self.sentiment_analyzer = SentimentAnalyzer(
            config['news_api_key'],
            config['twitter_api_key'],
            config['twitter_api_secret'],
            config['twitter_access_token'],
            config['twitter_access_token_secret']
        )
        self.advanced_news_analyzer = AdvancedNewsAnalyzer()
        self.ensemble_predictor = EnsemblePredictor()
        self.auto_model_selector = AutoModelSelector()
        self.pattern_detector = PatternDetector()
        self.trading_strategy = RLTradingStrategy()
        self.risk_manager = AdvancedRiskManager(config['initial_capital'])
        self.backtester = AdvancedBacktester(self.trading_strategy, self.risk_manager, 
                                             self.sentiment_analyzer)
        self.forward_tester = ForwardTester(self.trading_strategy, self.risk_manager, 
                                            self.sentiment_analyzer)
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            self.ensemble_predictor, self.trading_strategy, self.risk_manager,
            self.sentiment_analyzer
        )
        self.model_versioner = ModelVersioner()
        self.adaptation_counter = 0
        self.max_adaptations = 5  # Maximum number of adaptations before full retraining
        self.trading212_client = Trading212Client(
            config['trading212_username'],
            config['trading212_password'],
            config['chromedriver_path']
        )
        self.confidence_threshold = 0.90  # 90% confidence threshold for live trading

    # ... (rest of the class implementation remains the same)

def main():
    config = {
        'alpha_vantage_api_key': 'your_alpha_vantage_api_key',
        'news_api_key': 'your_news_api_key',
        'twitter_api_key': 'your_twitter_api_key',
        'twitter_api_secret': 'your_twitter_api_secret',
        'twitter_access_token': 'your_twitter_access_token',
        'twitter_access_token_secret': 'your_twitter_access_token_secret',
        'initial_capital': 100000,  # Initial capital for risk management
        'trading212_username': 'your_trading212_username',
        'trading212_password': 'your_trading212_password',
        'chromedriver_path': '/path/to/chromedriver'
    }

    trading_system = AdvancedGoldTradingSystem(config)

    # Run the system immediately
    trading_system.run()

    # Schedule the system to run daily at a specific time (e.g., 00:00 UTC)
    schedule.every().day.at("00:00").do(trading_system.run)

    while True:
        schedule.run_pending()
        time.sleep(60)  # Sleep for a minute before checking the schedule again

if __name__ == "__main__":
    main()

# You can add more main script logic here as needed