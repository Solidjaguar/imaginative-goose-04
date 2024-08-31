import pandas as pd
import numpy as np
from src.strategies.rl_trading_strategy import RLTradingStrategy
from src.risk_management.advanced_risk_manager import AdvancedRiskManager
from src.utils.sentiment_analyzer import SentimentAnalyzer
import logging

logger = logging.getLogger(__name__)

class AdvancedBacktester:
    def __init__(self, trading_strategy, risk_manager, sentiment_analyzer, commission_rate=0.001, slippage_pips=2):
        self.trading_strategy = trading_strategy
        self.risk_manager = risk_manager
        self.sentiment_analyzer = sentiment_analyzer
        self.results = None
        self.commission_rate = commission_rate  # 0.1% commission per trade
        self.slippage_pips = slippage_pips  # 2 pips slippage per trade

    def run_backtest(self, data, start_date, end_date):
        logger.info("Starting backtest...")
        
        # Filter data for the backtest period
        backtest_data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        # Initialize results DataFrame
        self.results = pd.DataFrame(index=backtest_data.index, columns=['Signal', 'Position', 'Price', 'Cash', 'Holdings', 'Equity', 'Returns'])
        
        position = 0
        entry_price = 0
        cash = self.risk_manager.initial_capital
        
        for i, (index, row) in enumerate(backtest_data.iterrows()):
            # Get trading signal
            signal = self.trading_strategy.generate_signal(row.values)
            
            # Apply risk management
            max_position_size = self.risk_manager.get_position_size(cash, row['Close'])
            
            # Get sentiment
            sentiment = self.sentiment_analyzer.get_sentiment('gold', index.strftime('%Y-%m-%d'))
            
            # Adjust signal based on sentiment (simple example, can be more sophisticated)
            if sentiment < -0.5 and signal == 1:  # Very negative sentiment, don't buy
                signal = 0
            elif sentiment > 0.5 and signal == -1:  # Very positive sentiment, don't sell
                signal = 0
            
            # Execute trades
            if signal == 1 and position <= 0:  # Buy signal
                trade_size = min(max_position_size, abs(position))
                slippage = self.slippage_pips * 0.0001 * row['Close']  # Convert pips to price
                execution_price = row['Close'] + slippage
                commission = trade_size * execution_price * self.commission_rate
                position += trade_size
                cash -= trade_size * execution_price + commission
                entry_price = execution_price
            elif signal == -1 and position >= 0:  # Sell signal
                trade_size = min(max_position_size, abs(position))
                slippage = self.slippage_pips * 0.0001 * row['Close']  # Convert pips to price
                execution_price = row['Close'] - slippage
                commission = trade_size * execution_price * self.commission_rate
                position -= trade_size
                cash += trade_size * execution_price - commission
                entry_price = execution_price
            
            # Calculate holdings and equity
            holdings = position * row['Close']
            equity = cash + holdings
            
            # Calculate returns
            returns = 0 if i == 0 else (equity - self.results.iloc[i-1]['Equity']) / self.results.iloc[i-1]['Equity']
            
            # Store results
            self.results.loc[index] = [signal, position, row['Close'], cash, holdings, equity, returns]
        
        logger.info("Backtest completed.")
        return self.results

    def calculate_metrics(self):
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        total_return = (self.results['Equity'].iloc[-1] - self.results['Equity'].iloc[0]) / self.results['Equity'].iloc[0]
        sharpe_ratio = np.sqrt(252) * self.results['Returns'].mean() / self.results['Returns'].std()
        max_drawdown = (self.results['Equity'] / self.results['Equity'].cummax() - 1).min()
        
        return {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }

    def get_best_trades(self, n=5):
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        trades = self.results[self.results['Signal'] != 0].copy()
        trades['Trade Return'] = trades['Returns'].shift(-1)
        best_trades = trades.nlargest(n, 'Trade Return')
        
        return best_trades[['Signal', 'Price', 'Trade Return']].to_dict('records')

# You can add more backtesting functionality here as needed