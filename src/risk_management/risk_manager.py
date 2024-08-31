import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, initial_capital, max_drawdown=0.2, var_threshold=0.05, position_size=0.1):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown = max_drawdown
        self.var_threshold = var_threshold
        self.position_size = position_size
        self.positions = {}

    def calculate_var(self, returns, confidence_level=0.95):
        mean = np.mean(returns)
        std = np.std(returns)
        var = norm.ppf(1 - confidence_level, mean, std)
        return abs(var)

    def calculate_position_size(self, price, volatility):
        return min(self.position_size * self.current_capital / price, 
                   self.current_capital * self.var_threshold / (price * volatility))

    def can_open_position(self, symbol, price, volatility):
        if symbol in self.positions:
            logger.warning(f"Position already open for {symbol}")
            return False
        
        position_size = self.calculate_position_size(price, volatility)
        var = price * volatility * position_size
        
        if var > self.var_threshold * self.current_capital:
            logger.warning(f"Opening position for {symbol} exceeds VaR threshold")
            return False
        
        return True

    def open_position(self, symbol, price, volatility):
        if self.can_open_position(symbol, price, volatility):
            position_size = self.calculate_position_size(price, volatility)
            cost = position_size * price
            self.positions[symbol] = {'size': position_size, 'price': price}
            self.current_capital -= cost
            logger.info(f"Opened position for {symbol}: {position_size} units at {price}")
            return True
        return False

    def close_position(self, symbol, price):
        if symbol in self.positions:
            position = self.positions[symbol]
            profit = (price - position['price']) * position['size']
            self.current_capital += (position['size'] * price)
            del self.positions[symbol]
            logger.info(f"Closed position for {symbol}: Profit/Loss: {profit}")
            return True
        logger.warning(f"No open position for {symbol}")
        return False

    def check_drawdown(self):
        drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        if drawdown > self.max_drawdown:
            logger.warning(f"Max drawdown exceeded: {drawdown:.2%}")
            return True
        return False

    def adjust_position_sizes(self):
        for symbol, position in self.positions.items():
            new_size = self.calculate_position_size(position['price'], 0.01)  # Assuming 1% daily volatility
            if new_size < position['size']:
                self.close_position(symbol, position['price'])
                self.open_position(symbol, position['price'], 0.01)
                logger.info(f"Adjusted position size for {symbol}")

    def get_risk_report(self):
        return {
            'current_capital': self.current_capital,
            'open_positions': self.positions,
            'drawdown': (self.initial_capital - self.current_capital) / self.initial_capital
        }

# You can add more risk management functions here as needed