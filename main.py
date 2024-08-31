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
from src.utils.liquidity_estimator import LiquidityEstimator
from src.optimization.portfolio_optimizer import PortfolioOptimizer
from src.models.execution_quality_predictor import ExecutionQualityPredictor
from src.visualization.advanced_visualizer import AdvancedVisualizer
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
        self.liquidity_estimator = LiquidityEstimator()
        self.portfolio_optimizer = PortfolioOptimizer(risk_free_rate=config['risk_free_rate'])
        self.execution_quality_predictor = ExecutionQualityPredictor()
        self.confidence_threshold = 0.90  # 90% confidence threshold for live trading
        self.performance_window = 30  # Number of days to consider for performance-based adjustments
        self.processed_data = None
        self.execution_history = pd.DataFrame(columns=['liquidity_estimate', 'actual_slippage', 'execution_time'])
        self.visualizer = AdvancedVisualizer(self)

    def run(self):
        logger.info("Starting advanced gold trading system...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # Get two years of data

        # Fetch and process data
        raw_data = self.data_fetcher.fetch_all_data(start_date, end_date)
        self.processed_data = self.feature_engineer.engineer_features(raw_data)

        # Train liquidity estimator
        self.liquidity_estimator.train(self.processed_data)

        # Estimate liquidity
        liquidity_estimate = self.liquidity_estimator.estimate_liquidity(self.processed_data.tail(1))
        logger.info(f"Current liquidity estimate: {liquidity_estimate}")

        # Train execution quality predictor
        self.execution_quality_predictor.train(self.processed_data)

        # Predict execution quality
        execution_quality = self.execution_quality_predictor.predict_execution_quality(self.processed_data.tail(1))
        logger.info(f"Predicted execution quality: {execution_quality}")

        # Calculate current volatility
        current_volatility = self.processed_data['Close'].pct_change().rolling(window=20).std().iloc[-1]

        # Detect drift
        if self.drift_detector.detect_drift(self.processed_data):
            logger.warning("Market drift detected. Adapting the system...")
            self.adapt_to_drift(self.processed_data)

        # Get sentiment data
        sentiment = self.sentiment_analyzer.get_combined_sentiment('gold', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        self.processed_data['sentiment'] = sentiment

        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(self.processed_data)
        self.processed_data = pd.concat([self.processed_data, patterns], axis=1)

        # Generate pattern-based signals
        pattern_signals = self.pattern_detector.generate_trading_signals(patterns, risk_tolerance='low')
        self.processed_data['pattern_signal'] = pattern_signals

        # Prepare data for prediction
        X = self.processed_data.drop(['Close'], axis=1)
        y = self.processed_data['Close']

        # Automatic model selection
        best_model, best_params = self.auto_model_selector.select_best_model(X, y)
        logger.info(f"Best model selected: {type(best_model).__name__}")
        logger.info(f"Best parameters: {best_params}")

        # Train ensemble predictor
        self.ensemble_predictor.train(X, y)

        # Optimize hyperparameters
        self.optimize_hyperparameters(self.processed_data)

        # Run backtesting
        backtest_results = self.backtester.run_backtest(self.processed_data, start_date, end_date - timedelta(days=30))
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
        forward_test_results = self.forward_tester.run_forward_test(self.processed_data, end_date - timedelta(days=30), end_date)
        forward_test_metrics = self.forward_tester.calculate_metrics()
        logger.info(f"Forward test metrics: {forward_test_metrics}")

        # Optimize portfolio
        returns = self.processed_data['Close'].pct_change().dropna()
        optimal_weights = self.portfolio_optimizer.optimize_sharpe_ratio(returns.to_frame('GOLD'))
        logger.info(f"Optimal portfolio weights: {optimal_weights}")

        # Make predictions and generate trading signal
        latest_data = self.processed_data.iloc[-1].to_frame().T
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
        self.adjust_trading_parameters(self.processed_data)

        # Execute live trade if confidence is above threshold
        if confidence >= self.confidence_threshold:
            self.execute_live_trade(final_signal, available_balance, current_volatility, self.processed_data['Close'].iloc[-1], liquidity_estimate, execution_quality)
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
        self.model_versioner.save_model(self.liquidity_estimator, 'liquidity_estimator')
        self.model_versioner.save_model(self.execution_quality_predictor, 'execution_quality_predictor')

        logger.info("Advanced trading cycle completed.")

    def execute_live_trade(self, signal, available_balance, current_volatility, current_price, liquidity_estimate, execution_quality):
        instrument = "GOLD"
        
        # Dynamically adjust slippage based on current market conditions and predicted execution quality
        adjusted_slippage = self.dynamic_adjuster.adjust_slippage(current_volatility, liquidity_estimate)
        predicted_slippage = execution_quality['predicted_slippage']
        final_slippage = (adjusted_slippage + predicted_slippage) / 2  # Average of both estimates
        
        # Measure and adjust internet delay
        measured_delay = self.dynamic_adjuster.measure_internet_delay()
        predicted_execution_time = execution_quality['predicted_execution_time']
        final_delay = max(measured_delay, predicted_execution_time)  # Use the larger of the two
        
        # Calculate position size based on current volatility, risk profile, and liquidity
        position_size = self.risk_manager.calculate_position_size(instrument, current_price, current_volatility)
        position_size = min(position_size, liquidity_estimate * 0.1)  # Limit position size to 10% of estimated liquidity
        
        if signal == 1:  # Buy signal
            # Simulate internet delay
            time.sleep(final_delay)
            
            # Fetch latest price to account for potential price changes during delay
            latest_price = self.trading212_client.get_latest_price(instrument)
            
            # Apply adjusted slippage
            execution_price = latest_price + (final_slippage * 0.0001 * latest_price)
            
            # Place the order
            order = self.trading212_client.place_order(instrument, position_size, "BUY", "MARKET")
            logger.info(f"Buy order placed: {order}")
            
            # Log the difference between expected and actual execution price
            actual_execution_price = order['executed_price']  # Assume this is provided by Trading212
            actual_slippage = (actual_execution_price - latest_price) / (0.0001 * latest_price)
            logger.info(f"Expected slippage: {final_slippage:.2f} pips, Actual slippage: {actual_slippage:.2f} pips")
            
            # Update risk manager
            self.risk_manager.update_position(instrument, position_size, actual_execution_price)
        
        elif signal == -1:  # Sell signal
            # Similar process for sell orders
            time.sleep(final_delay)
            latest_price = self.trading212_client.get_latest_price(instrument)
            execution_price = latest_price - (final_slippage * 0.0001 * latest_price)
            order = self.trading212_client.place_order(instrument, position_size, "SELL", "MARKET")
            logger.info(f"Sell order placed: {order}")
            actual_execution_price = order['executed_price']
            actual_slippage = (latest_price - actual_execution_price) / (0.0001 * latest_price)
            logger.info(f"Expected slippage: {final_slippage:.2f} pips, Actual slippage: {actual_slippage:.2f} pips")
            self.risk_manager.update_position(instrument, -position_size, actual_execution_price)
        
        else:
            logger.info("No trade signal. Holding current position.")

        # Update execution history
        self.execution_history = self.execution_history.append({
            'liquidity_estimate': liquidity_estimate,
            'actual_slippage': actual_slippage,
            'execution_time': final_delay
        }, ignore_index=True)

        # Update execution quality predictor with actual results
        self.execution_quality_predictor.update_model(latest_price, actual_execution_price, final_delay)

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
        'trading_frequency': 'medium',  # Can be 'low', 'medium', or 'high'
        'risk_free_rate': 0.02,  # 2% risk-free rate for portfolio optimization
    }

    trading_system = AdvancedGoldTradingSystem(config)

    # Run the system immediately
    trading_system.run()

    # Start the visualization dashboard
    trading_system.visualizer.run_dashboard()

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