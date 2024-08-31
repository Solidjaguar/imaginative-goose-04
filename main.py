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
        self.backtester = AdvancedBacktester(
            self.trading_strategy, 
            self.risk_manager, 
            self.sentiment_analyzer,
            commission_rate=config['commission_rate'],
            slippage_pips=config['slippage_pips']
        )
        self.forward_tester = ForwardTester(
            self.trading_strategy, 
            self.risk_manager, 
            self.sentiment_analyzer,
            commission_rate=config['commission_rate'],
            slippage_pips=config['slippage_pips']
        )
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
        self.commission_rate = config['commission_rate']
        self.slippage_pips = config['slippage_pips']
        self.internet_delay = config['internet_delay']  # in seconds

    def run(self):
        logger.info("Starting advanced gold trading system...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # Get two years of data

        # Fetch and process data
        raw_data = self.data_fetcher.fetch_all_data(start_date, end_date)
        processed_data = self.feature_engineer.engineer_features(raw_data)

        # ... (rest of the method remains the same until the trading signal generation)

        # Make predictions and generate trading signal
        latest_data = processed_data.iloc[-1].to_frame().T
        X_latest = latest_data.drop(['Close'], axis=1)
        
        ensemble_prediction = self.ensemble_predictor.predict(X_latest)
        best_model_prediction = best_model.predict(X_latest)
        
        # Combine predictions (you can adjust this logic based on your preference)
        final_prediction = (ensemble_prediction + best_model_prediction) / 2
        logger.info(f"Latest price prediction: {final_prediction[0]}")

        rl_signal = self.trading_strategy.generate_signal(latest_data.values[0])
        pattern_signal = pattern_signals.iloc[-1]
        
        # Combine signals (you can adjust this logic based on your preference)
        if rl_signal == pattern_signal:
            final_signal = rl_signal
        else:
            final_signal = 0  # Hold if signals disagree
        
        logger.info(f"Final trading signal: {final_signal}")

        # Calculate prediction confidence
        confidence = self.calculate_confidence(ensemble_prediction, best_model_prediction)
        logger.info(f"Prediction confidence: {confidence:.2f}")

        # Get account information
        account_info = self.trading212_client.get_account_info()
        available_balance = float(account_info['balance'].replace('$', '').replace(',', ''))
        logger.info(f"Available balance: ${available_balance:.2f}")

        # Execute live trade if confidence is above threshold
        if confidence >= self.confidence_threshold:
            self.execute_live_trade(final_signal, available_balance, current_price)
        else:
            logger.info(f"Confidence ({confidence:.2f}) below threshold. No trade executed.")

        # ... (rest of the method remains the same)

    def execute_live_trade(self, signal, available_balance, current_price):
        instrument = "GOLD"
        quantity = self.calculate_position_size(available_balance, current_price)
        
        if signal == 1:  # Buy signal
            # Simulate internet delay
            time.sleep(self.internet_delay)
            
            # Fetch latest price to account for potential price changes during delay
            latest_price = self.trading212_client.get_latest_price(instrument)
            
            # Apply slippage
            execution_price = latest_price + (self.slippage_pips * 0.0001 * latest_price)
            
            # Calculate commission
            commission = quantity * execution_price * self.commission_rate
            
            # Check if we have enough balance to execute the trade
            total_cost = (quantity * execution_price) + commission
            if total_cost > available_balance:
                logger.warning(f"Insufficient balance to execute buy order. Required: ${total_cost:.2f}, Available: ${available_balance:.2f}")
                return
            
            order = self.trading212_client.place_order(instrument, quantity, "BUY", "MARKET")
            logger.info(f"Buy order placed: {order}")
            logger.info(f"Estimated execution price: ${execution_price:.2f}")
            logger.info(f"Estimated commission: ${commission:.2f}")
        
        elif signal == -1:  # Sell signal
            # Simulate internet delay
            time.sleep(self.internet_delay)
            
            # Fetch latest price to account for potential price changes during delay
            latest_price = self.trading212_client.get_latest_price(instrument)
            
            # Apply slippage
            execution_price = latest_price - (self.slippage_pips * 0.0001 * latest_price)
            
            # Calculate commission
            commission = quantity * execution_price * self.commission_rate
            
            order = self.trading212_client.place_order(instrument, quantity, "SELL", "MARKET")
            logger.info(f"Sell order placed: {order}")
            logger.info(f"Estimated execution price: ${execution_price:.2f}")
            logger.info(f"Estimated commission: ${commission:.2f}")
        
        else:
            logger.info("No trade signal. Holding current position.")

    def calculate_position_size(self, available_balance, current_price):
        # Implement your position sizing logic here
        # This is a simple example, you should use a more sophisticated method
        max_risk_per_trade = 0.02  # 2% risk per trade
        position_size = (available_balance * max_risk_per_trade) / current_price
        return position_size

    # ... (rest of the class methods remain the same)

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
        'chromedriver_path': '/path/to/chromedriver',
        'commission_rate': 0.001,  # 0.1% commission per trade
        'slippage_pips': 2,  # 2 pips slippage per trade
        'internet_delay': 0.5  # 0.5 seconds internet delay
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