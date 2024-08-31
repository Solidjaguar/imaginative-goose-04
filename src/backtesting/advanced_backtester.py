import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import logging
from src.utils.sentiment_analyzer import SentimentAnalyzer
from src.utils.economic_indicators import EconomicIndicators

logger = logging.getLogger(__name__)

class AdvancedBacktester:
    def __init__(self, strategy, risk_manager, sentiment_analyzer, economic_indicators, initial_capital=100000):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.sentiment_analyzer = sentiment_analyzer
        self.economic_indicators = economic_indicators
        self.initial_capital = initial_capital
        self.results = defaultdict(list)

    def run_backtest(self, data, start_date, end_date):
        self.risk_manager.reset(self.initial_capital)
        portfolio_value = self.initial_capital
        position = 0

        for date, row in tqdm(data.loc[start_date:end_date].iterrows(), total=len(data.loc[start_date:end_date])):
            # Get sentiment and economic indicators for the current date
            sentiment = self.sentiment_analyzer.get_combined_sentiment('gold', date, date)
            indicators = self.economic_indicators.get_all_indicators(date, date)

            # Combine all features
            features = pd.concat([row.to_frame().T, pd.Series([sentiment], index=['sentiment']), indicators], axis=1)

            # Generate trading signal
            signal = self.strategy.generate_signal(features.values[0])

            # Apply risk management
            current_price = row['Close']
            volatility = data['Close'].pct_change().rolling(window=20).std().iloc[-1]

            if signal == 1 and position == 0:  # Buy signal
                if self.risk_manager.can_open_position('GOLD', current_price, volatility):
                    position = self.risk_manager.open_position('GOLD', current_price, volatility)
            elif signal == 2 and position != 0:  # Sell signal
                self.risk_manager.close_position('GOLD', current_price)
                position = 0

            # Update portfolio value
            portfolio_value = self.risk_manager.current_capital + (position * current_price)

            # Store results
            self.results['date'].append(date)
            self.results['portfolio_value'].append(portfolio_value)
            self.results['position'].append(position)
            self.results['price'].append(current_price)
            self.results['signal'].append(signal)

        return pd.DataFrame(self.results)

    def calculate_metrics(self):
        df = pd.DataFrame(self.results)
        df['returns'] = df['portfolio_value'].pct_change()

        total_return = (df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        sharpe_ratio = np.sqrt(252) * df['returns'].mean() / df['returns'].std()
        max_drawdown = (df['portfolio_value'] / df['portfolio_value'].cummax() - 1).min()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def get_best_trades(self, n=5):
        df = pd.DataFrame(self.results)
        df['trade_return'] = df['portfolio_value'].pct_change()
        df['trade_return'] = df['trade_return'].where(df['position'] != 0, 0)
        
        best_trades = df.nlargest(n, 'trade_return')
        return best_trades[['date', 'trade_return', 'price', 'signal']]

# You can add more backtesting methods here as needed