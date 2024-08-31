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
        self.trading212_client = Trading212Client(config['trading212_api_key'], config['trading212_account_id'])
        self.confidence_threshold = 0.90  # 90% confidence threshold for live trading

    def run(self):
        logger.info("Starting advanced gold trading system...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # Get two years of data

        # Fetch and process data
        raw_data = self.data_fetcher.fetch_all_data(start_date, end_date)
        processed_data = self.feature_engineer.engineer_features(raw_data)

        # Detect drift
        if self.drift_detector.detect_drift(processed_data):
            logger.warning("Market drift detected. Adapting the system...")
            self.adapt_to_drift(processed_data)

        # Get sentiment data
        sentiment = self.sentiment_analyzer.get_combined_sentiment('gold', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        processed_data['sentiment'] = sentiment

        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(processed_data)
        processed_data = pd.concat([processed_data, patterns], axis=1)

        # Generate pattern-based signals
        pattern_signals = self.pattern_detector.generate_trading_signals(patterns, risk_tolerance='low')
        processed_data['pattern_signal'] = pattern_signals

        # Prepare data for prediction
        X = processed_data.drop(['Close'], axis=1)
        y = processed_data['Close']

        # Automatic model selection
        best_model, best_params = self.auto_model_selector.select_best_model(X, y)
        logger.info(f"Best model selected: {type(best_model).__name__}")
        logger.info(f"Best parameters: {best_params}")

        # Train ensemble predictor
        self.ensemble_predictor.train(X, y)

        # Optimize hyperparameters
        self.optimize_hyperparameters(processed_data)

        # Run backtesting
        backtest_results = self.backtester.run_backtest(processed_data, start_date, end_date - timedelta(days=30))
        backtest_metrics = self.backtester.calculate_metrics()
        logger.info(f"Backtest metrics: {backtest_metrics}")

        # Get best trades from backtesting
        best_trades = self.backtester.get_best_trades(n=5)
        logger.info(f"Best trades from backtesting: {best_trades}")

        # Research and analyze news around best trades
        news_df = self.data_fetcher.fetch_news_for_dates([trade['date'] for trade in best_trades])
        analyzed_news = self.advanced_news_analyzer.analyze_news(news_df)
        news_summary = self.advanced_news_analyzer.summarize_analysis(analyzed_news)
        logger.info(f"Advanced news analysis summary: {news_summary}")

        # Run forward testing
        forward_test_results = self.forward_tester.run_forward_test(processed_data, end_date - timedelta(days=30), end_date)
        forward_test_metrics = self.forward_tester.calculate_metrics()
        logger.info(f"Forward test metrics: {forward_test_metrics}")

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
        available_balance = account_info['balance']
        logger.info(f"Available balance: {available_balance}")

        # Execute live trade if confidence is above threshold
        if confidence >= self.confidence_threshold:
            self.execute_live_trade(final_signal, available_balance, current_price)
        else:
            logger.info(f"Confidence ({confidence:.2f}) below threshold. No trade executed.")

        # Apply advanced risk management
        current_price = latest_data['Close'].values[0]
        volatility = processed_data['Close'].pct_change().rolling(window=20).std().iloc[-1]
        
        self.risk_manager.update_returns(processed_data['Close'].pct_change())

        # Check for max drawdown
        if self.risk_manager.check_drawdown():
            logger.warning("Max drawdown reached. Closing all positions.")
            self.close_all_positions()

        # Adjust position sizes based on current risk
        self.risk_manager.adjust_position_sizes()

        # Get risk report
        risk_report = self.risk_manager.get_risk_report()
        logger.info(f"Risk report: {risk_report}")

        # Save updated models
        self.model_versioner.save_model(self.ensemble_predictor, 'ensemble_model')
        self.model_versioner.save_model(best_model, 'best_model')
        self.model_versioner.save_model(self.trading_strategy, 'rl_model')

        logger.info("Advanced trading cycle completed.")

    def adapt_to_drift(self, data):
        logger.info("Adapting the system to market drift...")
        self.adaptation_counter += 1
        
        if self.adaptation_counter >= self.max_adaptations:
            logger.info("Maximum adaptations reached. Performing full retraining...")
            self.full_retrain(data)
        else:
            # Partial adaptation
            X = data.drop(['Close'], axis=1)
            y = data['Close']
            self.ensemble_predictor.update(X, y)
            self.auto_model_selector.update_best_model(X, y)
            self.trading_strategy.update(data)
            self.pattern_detector.update(data)
        
        logger.info("System adapted to new market conditions.")

    def full_retrain(self, data):
        logger.info("Performing full system retraining...")
        X = data.drop(['Close'], axis=1)
        y = data['Close']
        self.ensemble_predictor.train(X, y)
        self.auto_model_selector.select_best_model(X, y)
        self.trading_strategy.train(data)
        self.pattern_detector = PatternDetector()  # Reinitialize pattern detector
        self.adaptation_counter = 0
        logger.info("Full system retraining completed.")

    def optimize_hyperparameters(self, data):
        logger.info("Starting hyperparameter optimization...")
        
        # Optimize ensemble predictor
        X = data.drop(['Close'], axis=1)
        y = data['Close']
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.1, 0.2, 0.3]
        }
        self.ensemble_predictor = self.hyperparameter_optimizer.optimize_ensemble_predictor(X, y, param_distributions)

        # Optimize trading strategy
        self.hyperparameter_optimizer.optimize_trading_strategy(data)

        # Optimize risk manager
        self.hyperparameter_optimizer.optimize_risk_manager(data)

        logger.info("Hyperparameter optimization completed.")

    def calculate_confidence(self, ensemble_prediction, best_model_prediction):
        # This is a simple confidence calculation. You may want to use a more sophisticated method.
        difference = abs(ensemble_prediction - best_model_prediction)
        max_pred = max(ensemble_prediction, best_model_prediction)
        confidence = 1 - (difference / max_pred)
        return confidence[0]  # Return scalar value

    def execute_live_trade(self, signal, available_balance, current_price):
        instrument = "GOLD"
        quantity = self.calculate_position_size(available_balance, current_price)
        
        if signal == 1:  # Buy signal
            order = self.trading212_client.place_order(instrument, quantity, "BUY", "MARKET")
            logger.info(f"Buy order placed: {order}")
        elif signal == -1:  # Sell signal
            order = self.trading212_client.place_order(instrument, quantity, "SELL", "MARKET")
            logger.info(f"Sell order placed: {order}")
        else:
            logger.info("No trade signal. Holding current position.")

    def calculate_position_size(self, available_balance, current_price):
        # Implement your position sizing logic here
        # This is a simple example, you should use a more sophisticated method
        max_risk_per_trade = 0.02  # 2% risk per trade
        position_size = (available_balance * max_risk_per_trade) / current_price
        return position_size

    def close_all_positions(self):
        positions = self.trading212_client.get_positions()
        for position in positions:
            self.trading212_client.place_order(
                position['instrument'],
                position['quantity'],
                "SELL" if position['side'] == "BUY" else "BUY",
                "MARKET"
            )
        logger.info("All positions closed.")

def main():
    config = {
        'alpha_vantage_api_key': 'your_alpha_vantage_api_key',
        'news_api_key': 'your_news_api_key',
        'twitter_api_key': 'your_twitter_api_key',
        'twitter_api_secret': 'your_twitter_api_secret',
        'twitter_access_token': 'your_twitter_access_token',
        'twitter_access_token_secret': 'your_twitter_access_token_secret',
        'initial_capital': 100000,  # Initial capital for risk management
        'trading212_api_key': 'your_trading212_api_key',
        'trading212_account_id': 'your_trading212_account_id'
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