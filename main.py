import pandas as pd
import numpy as np
from src.utils.data_fetcher import DataFetcher
from src.utils.data_processor import DataProcessor
from src.utils.drift_detector import DriftDetector
from src.utils.feature_analyzer import FeatureAnalyzer
from src.models.gold_price_predictor import GoldPricePredictor
from src.strategies.rl_trading_strategy import RLTradingStrategy
from src.backtesting.backtester import Backtester
import logging
import schedule
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedGoldTradingSystem:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.data_processor = DataProcessor()
        self.drift_detector = DriftDetector()
        self.feature_analyzer = FeatureAnalyzer()
        self.price_predictor = GoldPricePredictor()
        self.trading_strategy = RLTradingStrategy()
        self.backtester = Backtester(self.trading_strategy)

    def run(self):
        logger.info("Starting automated gold trading system...")

        # Fetch and process data
        raw_data = self.data_fetcher.get_data()
        processed_data = self.data_processor.process_data(raw_data)

        # Detect drift
        if self.drift_detector.detect_drift(processed_data):
            logger.warning("Market drift detected. Adapting the system...")
            self.adapt_to_drift(processed_data)

        # Analyze and engineer features
        self.feature_analyzer.analyze_features(processed_data.drop('target', axis=1), processed_data['target'])
        selected_features = self.feature_analyzer.select_features(processed_data.drop('target', axis=1))
        engineered_data = self.feature_analyzer.engineer_features(selected_features)
        engineered_data['target'] = processed_data['target']

        # Train or update price predictor
        try:
            self.price_predictor.load_model('best_model.joblib')
            logger.info("Loaded existing price predictor model.")
        except:
            logger.info("No existing price predictor model found. Training a new model.")
            self.price_predictor.train(engineered_data)

        self.price_predictor.update(engineered_data)

        # Train or update RL trading strategy
        try:
            self.trading_strategy.load_model('rl_model.zip')
            logger.info("Loaded existing RL trading strategy model.")
        except:
            logger.info("No existing RL trading strategy model found. Training a new model.")
            self.trading_strategy.train(engineered_data)

        # Make predictions and generate trading signal
        latest_data = engineered_data.iloc[-1].drop('target').to_frame().T
        prediction = self.price_predictor.predict(latest_data)
        logger.info(f"Latest price prediction: {prediction[0]}")

        rl_observation = latest_data.values[0]
        signal = self.trading_strategy.generate_signal(rl_observation)
        logger.info(f"RL Trading signal: {signal}")

        # Backtesting
        backtest_results = self.backtester.run_backtest(engineered_data)
        logger.info(f"Backtest results: {backtest_results}")

        # Here you would typically execute the trade based on the signal
        # This part would involve integrating with a broker's API
        logger.info("Trade execution would happen here in a live system.")

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

def main():
    trading_system = AutomatedGoldTradingSystem()

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