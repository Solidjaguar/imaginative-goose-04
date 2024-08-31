import pandas as pd
import numpy as np
from src.utils.data_fetcher import DataFetcher
from src.utils.data_processor import DataProcessor
from src.models.gold_price_predictor import GoldPricePredictor
from src.strategies.trading_strategy import TradingStrategy
from src.backtesting.backtesting import Backtester
import logging
import schedule
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedGoldTradingSystem:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.data_processor = DataProcessor()
        self.model = GoldPricePredictor()
        self.strategy = TradingStrategy(self.model)
        self.backtester = Backtester(self.strategy)

    def run(self):
        logger.info("Starting automated gold trading system...")

        # Fetch and process data
        raw_data = self.data_fetcher.get_data()
        processed_data = self.data_processor.process_data(raw_data)

        # Check if model exists, if not, train a new one
        try:
            self.model.load_model('best_model.joblib')
            logger.info("Loaded existing model.")
        except:
            logger.info("No existing model found. Training a new model.")
            self.model.train(processed_data)
            self.model.save_model('best_model.joblib')

        # Update model with new data
        self.model.update(processed_data)

        # Make predictions
        latest_data = processed_data.iloc[-1].drop('target').to_frame().T
        prediction = self.model.predict(latest_data)
        logger.info(f"Latest prediction: {prediction[0]}")

        # Apply trading strategy
        signal = self.strategy.generate_signal(latest_data)
        logger.info(f"Trading signal: {signal}")

        # Backtesting
        backtest_results = self.backtester.run_backtest(processed_data)
        logger.info(f"Backtest results: {backtest_results}")

        # Here you would typically execute the trade based on the signal
        # This part would involve integrating with a broker's API
        logger.info("Trade execution would happen here in a live system.")

        # Save updated model
        self.model.save_model('best_model.joblib')

        logger.info("Automated trading cycle completed.")

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