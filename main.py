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
from src.utils.dynamic_trade_adjuster import DynamicTradeAdjuster
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
        self.backtester = AdvancedBacktester(self.trading_strategy, self.risk_manager, self.sentiment_analyzer)
        self.forward_tester = ForwardTester(self.trading_strategy, self.risk_manager, self.sentiment_analyzer)
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
        self.dynamic_adjuster = DynamicTradeAdjuster(
            initial_slippage_pips=config['initial_slippage_pips'],
            initial_internet_delay=config['initial_internet_delay']
        )
        self.confidence_threshold = 0.90  # 90% confidence threshold for live trading
        self.performance_window = 30  # Number of days to consider for performance-based adjustments

    def run(self):
        logger.info("Starting advanced gold trading system...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # Get two years of data

        # Fetch and process data
        raw_data = self.data_fetcher.fetch_all_data(start_date, end_date)
        processed_data = self.feature_engineer.engineer_features(raw_data)

        # Calculate current volatility
        current_volatility = processed_data['Close'].pct_change().rolling(window=20).std().iloc[-1]

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
        available_balance = float(account_info['balance'].replace('$', '').replace(',', ''))
        logger.info(f"Available balance: ${available_balance:.2f}")

        # Adjust trading parameters based on recent performance
        self.adjust_trading_parameters(processed_data)

        # Execute live trade if confidence is above threshold
        if confidence >= self.confidence_threshold:
            self.execute_live_trade(final_signal, available_balance, current_volatility, processed_data['Close'].iloc[-1])
        else:
            logger.info(f"Confidence ({confidence:.2f}) below threshold. No trade executed.")

        # Apply advanced risk management
        if not self.risk_manager.check_risk_limits():
            logger.warning("Risk limits exceeded. Reducing positions.")
            self.reduce_positions()

        # Get risk report
        risk_report = self.risk_manager.get_risk_report()
        logger.info(f"Risk report: {risk_report}")

        # Save updated models
        self.model_versioner.save_model(self.ensemble_predictor, 'ensemble_model')
        self.model_versioner.save_model(best_model, 'best_model')
        self.model_versioner.save_model(self.trading_strategy, 'rl_model')

        logger.info("Advanced trading cycle completed.")

    def execute_live_trade(self, signal, available_balance, current_volatility, current_price):
        instrument = "GOLD"
        
        # Dynamically adjust slippage based on current market conditions
        current_liquidity = self.estimate_liquidity()  # You need to implement this method
        adjusted_slippage = self.dynamic_adjuster.adjust_slippage(current_volatility, current_liquidity)
        
        # Measure and adjust internet delay
        measured_delay = self.dynamic_adjuster.measure_internet_delay()
        
        # Calculate position size based on current volatility and risk profile
        position_size = self.risk_manager.calculate_position_size(instrument, current_price, current_volatility)
        
        if signal == 1:  # Buy signal
            # Simulate internet delay
            time.sleep(measured_delay)
            
            # Fetch latest price to account for potential price changes during delay
            latest_price = self.trading212_client.get_latest_price(instrument)
            
            # Apply adjusted slippage
            execution_price = latest_price + (adjusted_slippage * 0.0001 * latest_price)
            
            # Place the order
            order = self.trading212_client.place_order(instrument, position_size, "BUY", "MARKET")
            logger.info(f"Buy order placed: {order}")
            
            # Log the difference between expected and actual execution price
            actual_execution_price = order['executed_price']  # Assume this is provided by Trading212
            avg_slippage = self.dynamic_adjuster.log_slippage(execution_price, actual_execution_price)
            logger.info(f"Average slippage: {avg_slippage:.2f} pips")
            
            # Update risk manager
            self.risk_manager.update_position(instrument, position_size, actual_execution_price)
        
        elif signal == -1:  # Sell signal
            # Similar process for sell orders
            time.sleep(measured_delay)
            latest_price = self.trading212_client.get_latest_price(instrument)
            execution_price = latest_price - (adjusted_slippage * 0.0001 * latest_price)
            order = self.trading212_client.place_order(instrument, position_size, "SELL", "MARKET")
            logger.info(f"Sell order placed: {order}")
            actual_execution_price = order['executed_price']
            avg_slippage = self.dynamic_adjuster.log_slippage(execution_price, actual_execution_price)
            logger.info(f"Average slippage: {avg_slippage:.2f} pips")
            self.risk_manager.update_position(instrument, -position_size, actual_execution_price)
        
        else:
            logger.info("No trade signal. Holding current position.")

    def adjust_trading_parameters(self, data):
        # Calculate recent performance
        recent_returns = data['Close'].pct_change().tail(self.performance_window)
        recent_sharpe = np.sqrt(252) * recent_returns.mean() / recent_returns.std()
        
        # Adjust confidence threshold based on recent performance
        if recent_sharpe > 1.5:
            self.confidence_threshold = max(0.85, self.confidence_threshold - 0.01)
        elif recent_sharpe < 0.5:
            self.confidence_threshold = min(0.95, self.confidence_threshold + 0.01)
        
        # Adjust trading frequency
        if recent_sharpe > 2:
            self.config['trading_frequency'] = 'high'
        elif recent_sharpe < 0:
            self.config['trading_frequency'] = 'low'
        else:
            self.config['trading_frequency'] = 'medium'
        
        logger.info(f"Adjusted confidence threshold: {self.confidence_threshold:.2f}")
        logger.info(f"Adjusted trading frequency: {self.config['trading_frequency']}")

    def reduce_positions(self):
        for instrument, position in self.risk_manager.positions.items():
            if position['size'] > 0:
                reduction_size = position['size'] * 0.5  # Reduce position by 50%
                order = self.trading212_client.place_order(instrument, reduction_size, "SELL", "MARKET")
                logger.info(f"Reducing position: Sold {reduction_size} of {instrument}")
                self.risk_manager.update_position(instrument, -reduction_size, order['executed_price'])

    def estimate_liquidity(self):
        # This is a placeholder. In a real implementation, you would estimate liquidity
        # based on factors like trading volume, bid-ask spread, etc.
        return 1.0

    # ... (other methods remain the same)

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
        'initial_slippage_pips': 2,
        'initial_internet_delay': 0.5,
        'trading_frequency': 'medium'  # Can be 'low', 'medium', or 'high'
    }

    trading_system = AdvancedGoldTradingSystem(config)

    # Run the system immediately
    trading_system.run()

    # Schedule the system to run based on trading frequency
    if config['trading_frequency'] == 'high':
        schedule.every(1).hours.do(trading_system.run)
    elif config['trading_frequency'] == 'medium':
        schedule.every(4).hours.do(trading_system.run)
    else:  # low frequency
        schedule.every().day.at("00:00").do(trading_system.run)

    while True:
        schedule.run_pending()
        time.sleep(60)  # Sleep for a minute before checking the schedule again

if __name__ == "__main__":
    main()

# You can add more main script logic here as needed