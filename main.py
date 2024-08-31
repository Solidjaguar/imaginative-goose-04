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

def run_trading_system():
    logger.info("Starting automated gold trading system...")

    # Fetch and process data
    data_fetcher = DataFetcher()
    raw_data = data_fetcher.get_data()

    data_processor = DataProcessor()
    processed_data = data_processor.process_data(raw_data)

    # Train and save the model
    model = GoldPricePredictor()
    model.train(processed_data)
    model.save_model('best_model.joblib')

    # Make predictions
    latest_data = processed_data.iloc[-1].drop('target').to_frame().T
    prediction = model.predict(latest_data)
    logger.info(f"Latest prediction: {prediction[0]}")

    # Apply trading strategy
    strategy = TradingStrategy(model)
    signal = strategy.generate_signal(latest_data)
    logger.info(f"Trading signal: {signal}")

    # Backtesting
    backtester = Backtester(strategy)
    backtest_results = backtester.run_backtest(processed_data)
    logger.info(f"Backtest results: {backtest_results}")

    # Here you would typically execute the trade based on the signal
    # This part would involve integrating with a broker's API
    logger.info("Trade execution would happen here in a live system.")

    logger.info("Automated trading cycle completed.")

def main():
    # Run the system immediately
    run_trading_system()

    # Schedule the system to run daily at a specific time (e.g., 00:00 UTC)
    schedule.every().day.at("00:00").do(run_trading_system)

    while True:
        schedule.run_pending()
        time.sleep(60)  # Sleep for a minute before checking the schedule again

if __name__ == "__main__":
    main()

# You can add more main script logic here as needed