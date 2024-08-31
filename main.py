import pandas as pd
import numpy as np
from src.utils.data_fetcher import DataFetcher
from src.utils.data_processor import DataProcessor
from src.utils.drift_detector import DriftDetector
from src.utils.feature_analyzer import FeatureAnalyzer
from src.utils.sentiment_analyzer import SentimentAnalyzer
from src.utils.economic_indicators import EconomicIndicators
from src.utils.news_researcher import NewsResearcher
from src.utils.advanced_news_analyzer import AdvancedNewsAnalyzer
from src.models.gold_price_predictor import GoldPricePredictor
from src.strategies.rl_trading_strategy import RLTradingStrategy
from src.backtesting.advanced_backtester import AdvancedBacktester
from src.forward_testing.forward_tester import ForwardTester
from src.risk_management.risk_manager import RiskManager
from src.optimization.hyperparameter_optimizer import HyperparameterOptimizer
import logging
import schedule
import time
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedGoldTradingSystem:
    def __init__(self, config):
        self.config = config
        self.data_fetcher = DataFetcher()
        self.data_processor = DataProcessor()
        self.drift_detector = DriftDetector()
        self.feature_analyzer = FeatureAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer(
            config['news_api_key'],
            config['twitter_api_key'],
            config['twitter_api_secret'],
            config['twitter_access_token'],
            config['twitter_access_token_secret']
        )
        self.economic_indicators = EconomicIndicators(config['fred_api_key'])
        self.news_researcher = NewsResearcher(config['news_api_key'])
        self.advanced_news_analyzer = AdvancedNewsAnalyzer()
        self.price_predictor = GoldPricePredictor()
        self.trading_strategy = RLTradingStrategy()
        self.risk_manager = RiskManager(config['initial_capital'])
        self.backtester = AdvancedBacktester(self.trading_strategy, self.risk_manager, 
                                             self.sentiment_analyzer, self.economic_indicators)
        self.forward_tester = ForwardTester(self.trading_strategy, self.risk_manager, 
                                            self.sentiment_analyzer, self.economic_indicators)
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            self.price_predictor, self.trading_strategy, self.risk_manager,
            self.sentiment_analyzer, self.economic_indicators
        )

    def run(self):
        logger.info("Starting automated gold trading system...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # Get two years of data

        # Fetch and process data
        raw_data = self.data_fetcher.get_data(start_date, end_date)
        processed_data = self.data_processor.process_data(raw_data)

        # Detect drift
        if self.drift_detector.detect_drift(processed_data):
            logger.warning("Market drift detected. Adapting the system...")
            self.adapt_to_drift(processed_data)

        # Get sentiment data and economic indicators
        sentiment = self.sentiment_analyzer.get_combined_sentiment('gold', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        indicators = self.economic_indicators.get_all_indicators(start_date, end_date)

        # Combine all data
        full_data = pd.concat([processed_data, pd.Series(sentiment, name='sentiment'), indicators], axis=1)

        # Optimize hyperparameters
        self.optimize_hyperparameters(full_data)

        # Run backtesting
        backtest_results = self.backtester.run_backtest(full_data, start_date, end_date - timedelta(days=30))
        backtest_metrics = self.backtester.calculate_metrics()
        logger.info(f"Backtest metrics: {backtest_metrics}")

        # Get best trades from backtesting
        best_trades = self.backtester.get_best_trades(n=5)
        logger.info(f"Best trades from backtesting: {best_trades}")

        # Research and analyze news around best trades
        news_df = self.news_researcher.analyze_news_for_trades(best_trades)
        analyzed_news = self.advanced_news_analyzer.analyze_news(news_df)
        news_summary = self.advanced_news_analyzer.summarize_analysis(analyzed_news)
        logger.info(f"Advanced news analysis summary: {news_summary}")

        # Run forward testing
        forward_test_results = self.forward_tester.run_forward_test(full_data, end_date - timedelta(days=30), end_date)
        forward_test_metrics = self.forward_tester.calculate_metrics()
        logger.info(f"Forward test metrics: {forward_test_metrics}")

        # Train or update price predictor
        self.price_predictor.train(full_data)

        # Make predictions and generate trading signal
        latest_data = full_data.iloc[-1].to_frame().T
        prediction = self.price_predictor.predict(latest_data)
        logger.info(f"Latest price prediction: {prediction[0]}")

        signal = self.trading_strategy.generate_signal(latest_data.values[0])
        logger.info(f"RL Trading signal: {signal}")

        # Apply risk management
        current_price = latest_data['Close'].values[0]
        volatility = full_data['Close'].pct_change().std()
        
        if signal == 1:  # Buy signal
            if self.risk_manager.can_open_position('GOLD', current_price, volatility):
                self.risk_manager.open_position('GOLD', current_price, volatility)
        elif signal == 2:  # Sell signal
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

        logger.info("Automated trading cycle completed.")

    def adapt_to_drift(self, data):
        logger.info("Adapting the system to market drift...")
        # Retrain models from scratch
        self.price_predictor.train(data, force=True)
        self.trading_strategy.train(data)
        logger.info("System adapted to new market conditions.")

    def optimize_hyperparameters(self, data):
        logger.info("Starting hyperparameter optimization...")
        
        # Optimize price predictor
        X = data.drop('target', axis=1)
        y = data['target']
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
        'news_api_key': 'your_news_api_key',
        'twitter_api_key': 'your_twitter_api_key',
        'twitter_api_secret': 'your_twitter_api_secret',
        'twitter_access_token': 'your_twitter_access_token',
        'twitter_access_token_secret': 'your_twitter_access_token_secret',
        'fred_api_key': 'your_fred_api_key',
        'initial_capital': 100000  # Initial capital for risk management
    }

    trading_system = AutomatedGoldTradingSystem(config)

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