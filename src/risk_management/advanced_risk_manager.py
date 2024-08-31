import numpy as np
import pandas as pd
from scipy.stats import norm

class AdvancedRiskManager:
    def __init__(self, initial_capital, max_portfolio_risk=0.02, max_drawdown=0.1):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown = max_drawdown
        self.positions = {}
        self.returns_history = []

    def update_capital(self, new_capital):
        self.current_capital = new_capital

    def update_returns(self, returns):
        self.returns_history.append(returns)
        if len(self.returns_history) > 252:  # Keeping approximately 1 year of daily returns
            self.returns_history.pop(0)

    def calculate_position_size(self, instrument, current_price, volatility):
        # Calculate position size based on volatility and portfolio risk
        portfolio_value = self.current_capital + sum(pos['value'] for pos in self.positions.values())
        risk_amount = portfolio_value * self.max_portfolio_risk
        position_size = risk_amount / (volatility * current_price)
        
        # Adjust position size based on current portfolio composition
        total_exposure = sum(abs(pos['size']) for pos in self.positions.values())
        if total_exposure > 0:
            position_size *= (1 - total_exposure / portfolio_value)
        
        return max(0, position_size)  # Ensure non-negative position size

    def update_position(self, instrument, size, price):
        if instrument in self.positions:
            self.positions[instrument]['size'] += size
            self.positions[instrument]['value'] = self.positions[instrument]['size'] * price
        else:
            self.positions[instrument] = {'size': size, 'value': size * price}

    def calculate_portfolio_var(self, confidence_level=0.95):
        if not self.returns_history:
            return 0
        
        returns = np.array(self.returns_history)
        portfolio_value = self.current_capital + sum(pos['value'] for pos in self.positions.values())
        var = norm.ppf(1 - confidence_level) * np.std(returns) * np.sqrt(252) * portfolio_value
        return abs(var)

    def check_risk_limits(self):
        portfolio_var = self.calculate_portfolio_var()
        portfolio_value = self.current_capital + sum(pos['value'] for pos in self.positions.values())
        
        if portfolio_var > self.max_portfolio_risk * portfolio_value:
            return False
        
        drawdown = (portfolio_value - self.initial_capital) / self.initial_capital
        if drawdown < -self.max_drawdown:
            return False
        
        return True

    def get_risk_report(self):
        portfolio_value = self.current_capital + sum(pos['value'] for pos in self.positions.values())
        portfolio_var = self.calculate_portfolio_var()
        drawdown = (portfolio_value - self.initial_capital) / self.initial_capital
        
        return {
            'Portfolio Value': portfolio_value,
            'Value at Risk (95%)': portfolio_var,
            'Current Drawdown': drawdown,
            'Positions': self.positions
        }

# You can add more risk management functionality here as needed