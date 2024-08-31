import pandas as pd
import numpy as np
from src.utils.advanced_data_fetcher import AdvancedDataFetcher
from src.utils.advanced_feature_engineering import AdvancedFeatureEngineer
from src.utils.drift_detector import DriftDetector
from src.utils.sentiment_analyzer import SentimentAnalyzer
from src.utils.advanced_news_analyzer import AdvancedNewsAnalyzer
from src.models.gold_price_predictor import GoldPricePredictor
from src.models.pattern_detector import PatternDetector
from src.strategies.rl_trading_strategy import RLTradingStrategy
from src.backtesting.advanced_backtester import AdvancedBacktester
from src.forward_testing.forward_tester import ForwardTester
from src.risk_management.advanced_risk_manager import AdvancedRiskManager
from src.optimization.hyperparameter_optimizer import HyperparameterOptimizer
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
        self.price_predictor = GoldPricePredictor()
        self.pattern_detector = PatternDetector()
        self.trading_strategy = RLTradingStrategy()
        self.risk_manager = AdvancedRiskManager(config['initial_capital'])
        self.backtester = AdvancedBacktester(self.trading_strategy, self.risk_manager, 
                                             self.sentiment_analyzer)
        self.forward_tester = ForwardTester(self.trading_strategy, self.risk_manager, 
                                            self.sentiment_analyzer)
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            self.price_predictor, self.trading_strategy, self.risk_manager,
            self.sentiment_analyzer
        )

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

        # Train or update price predictor
        self.price_predictor.train(processed_data)

        # Make predictions and generate trading signal
        latest_data = processed_data.iloc[-1].to_frame().T
        prediction = self.price_predictor.predict(latest_data)
        logger.info(f"Latest price prediction: {prediction[0]}")

        rl_signal = self.trading_strategy.generate_signal(latest_data.values[0])
        pattern_signal = pattern_signals.iloc[-1]
        
        # Combine signals (you can adjust this logic based on your preference)
        if rl_signal == pattern_signal:
            final_signal = rl_signal
        else:
            final_signal = 0  # Hold if signals disagree
        
        logger.info(f"Final trading signal: {final_signal}")

        # Apply advanced risk management
        current_price = latest_data['Close'].values[0]
        volatility = processed_data['Close'].pct_change().rolling(window=20).std().iloc[-1]
        
        self.risk_manager.update_returns(processed_data['Close'].pct_change())

        if final_signal == 1:  # Buy signal
            self.risk_manager.open_position('GOLD', current_price, volatility)
        elif final_signal == -1:  # Sell signal
            self.risk_manager.close_position('GOLD', current_price)

        # Check for max drawdown
        if self.risk_manager.check_drawdown():
            logger.warning("Max drawdown reached. Closing all positions.")
            self.risk_manager.close_position('GOLD', current_price)

        # Adjust position sizes based on current risk
        self.risk_manager.adjust_position_sizes()

        # Get risk report
        risk_report = self.risk_manager.get_risk_report()
        logger.info(f"Risk report: {risk_report}")

        # Save updated models
        self.price_predictor.save_model('best_model.joblib')
        self.trading_strategy.save_model('rl_model.zip')

        logger.info("Advanced trading cycle completed.")

    def adapt_to_drift(self, data):
        logger.info("Adapting the system to market drift...")
        # Retrain models from scratch
        self.price_predictor.train(data, force=True)
        self.trading_strategy.train(data)
        self.pattern_detector = PatternDetector()  # Reinitialize pattern detector
        logger.info("System adapted to new market conditions.")

    def optimize_hyperparameters(self, data):
        logger.info("Starting hyperparameter optimization...")
        
        # Optimize price predictor
        X = data.drop('Close', axis=1)
        y = data['Close']
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.1, 0.2, 0.3]
        }
        self.price_predictor = self.hyperparameter_optimizer.optimize_price_predictor(X, y, param_distributions)

        # Optimize trading strategy
        self.hyperparameter_optimizer.optimize_trading_strategy(data)

        # Optimize risk manager
        self.hyperparameter_optimizer.optimize_risk_manager(data)

        logger.info("Hyperparameter optimization completed.")

def main():
    config = {
        'alpha_vantage_api_key': 'your_alpha_vantage_api_key',
        'news_api_key': 'your_news_api_key',
        'twitter_api_key': 'your_twitter_api_key',
        'twitter_api_secret': 'your_twitter_api_secret',
        'twitter_access_token': 'your_twitter_access_token',
        'twitter_access_token_secret': 'your_twitter_access_token_secret',
        'initial_capital': 100000  # Initial capital for risk management
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