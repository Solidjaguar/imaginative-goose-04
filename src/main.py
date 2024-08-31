import pandas as pd
from data_handler import fetch_gold_data, preprocess_data
from ai_model import GoldTradingAI
from backtesting import backtest, calculate_performance_metrics

def main():
    # Fetch and preprocess data
    start_date = "2010-01-01"
    end_date = "2023-05-23"  # Current date
    raw_data = fetch_gold_data(start_date, end_date)
    processed_data = preprocess_data(raw_data)
    
    # Initialize and train the AI model
    ai_model = GoldTradingAI(lookback=30)
    ai_model.train(processed_data)
    
    # Generate trading signals
    signals = ai_model.generate_signals(processed_data)
    
    # Perform backtesting
    backtest_results = backtest(processed_data, lambda data: signals)
    performance_metrics = calculate_performance_metrics(backtest_results)
    
    # Print results
    print("Backtesting Results:")
    print(f"Total Return: {performance_metrics['Total Return']:.2%}")
    print(f"Sharpe Ratio: {performance_metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {performance_metrics['Max Drawdown']:.2%}")
    
    # Generate latest trading signal
    latest_data = processed_data.tail(31)  # 30 for lookback + 1 for current day
    latest_signal = ai_model.predict(latest_data)
    print(f"\nLatest Trading Signal: {'BUY' if latest_signal > 0.5 else 'SELL'}")

if __name__ == "__main__":
    main()

# You can add more main script logic here as needed