import numpy as np
import time

class DynamicTradeAdjuster:
    def __init__(self, initial_slippage_pips=2, initial_internet_delay=0.5):
        self.slippage_pips = initial_slippage_pips
        self.internet_delay = initial_internet_delay
        self.slippage_history = []
        self.delay_history = []

    def adjust_slippage(self, current_volatility, current_liquidity):
        # Adjust slippage based on current market conditions
        volatility_factor = current_volatility / np.mean(self.slippage_history[-30:]) if self.slippage_history else 1
        liquidity_factor = 1 / current_liquidity  # Assume higher liquidity leads to lower slippage
        
        new_slippage = self.slippage_pips * volatility_factor * liquidity_factor
        self.slippage_pips = max(1, min(10, new_slippage))  # Keep slippage between 1 and 10 pips
        
        return self.slippage_pips

    def measure_internet_delay(self):
        start_time = time.time()
        # Simulate a ping to Trading212 server
        # In a real implementation, you would make a lightweight API call to Trading212
        time.sleep(0.1)  # Simulating network latency
        end_time = time.time()
        
        measured_delay = end_time - start_time
        self.delay_history.append(measured_delay)
        
        if len(self.delay_history) > 100:
            self.delay_history.pop(0)
        
        self.internet_delay = np.mean(self.delay_history)
        return self.internet_delay

    def log_slippage(self, expected_price, actual_price):
        actual_slippage = abs(actual_price - expected_price) / (0.0001 * expected_price)  # Convert to pips
        self.slippage_history.append(actual_slippage)
        
        if len(self.slippage_history) > 100:
            self.slippage_history.pop(0)
        
        return np.mean(self.slippage_history)

    def get_current_slippage(self):
        return self.slippage_pips

    def get_current_delay(self):
        return self.internet_delay

# You can add more dynamic adjustment functionality here as needed