from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import logging

logger = logging.getLogger(__name__)

class Trading212Client:
    def __init__(self, username, password, chromedriver_path):
        self.username = username
        self.password = password
        self.driver = webdriver.Chrome(executable_path=chromedriver_path)
        self.logged_in = False

    def login(self):
        try:
            self.driver.get("https://www.trading212.com/en/login")
            
            # Wait for the username field to be visible
            username_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "username-real"))
            )
            username_field.send_keys(self.username)
            
            # Find and fill in the password field
            password_field = self.driver.find_element(By.ID, "pass-real")
            password_field.send_keys(self.password)
            
            # Click the login button
            login_button = self.driver.find_element(By.CLASS_NAME, "button-login")
            login_button.click()
            
            # Wait for the dashboard to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "dashboard"))
            )
            
            self.logged_in = True
            logger.info("Successfully logged in to Trading212")
        except TimeoutException:
            logger.error("Timeout while trying to log in to Trading212")
        except Exception as e:
            logger.error(f"Error logging in to Trading212: {str(e)}")

    def get_account_info(self):
        if not self.logged_in:
            self.login()
        
        try:
            # Navigate to the account page
            self.driver.get("https://www.trading212.com/en/account")
            
            # Wait for the account balance to be visible
            balance_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "account-balance"))
            )
            
            balance = balance_element.text
            
            return {"balance": balance}
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None

    def place_order(self, instrument, quantity, side, type, limit_price=None):
        if not self.logged_in:
            self.login()
        
        try:
            # Navigate to the trading page for the instrument
            self.driver.get(f"https://www.trading212.com/en/trading/{instrument}")
            
            # Wait for the order form to be visible
            order_form = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "order-form"))
            )
            
            # Fill in the quantity
            quantity_field = order_form.find_element(By.NAME, "quantity")
            quantity_field.clear()
            quantity_field.send_keys(str(quantity))
            
            # Select the order type
            type_dropdown = order_form.find_element(By.NAME, "orderType")
            type_dropdown.send_keys(type)
            
            if type == "LIMIT" and limit_price:
                limit_price_field = order_form.find_element(By.NAME, "limitPrice")
                limit_price_field.clear()
                limit_price_field.send_keys(str(limit_price))
            
            # Click the Buy or Sell button
            if side == "BUY":
                buy_button = order_form.find_element(By.CLASS_NAME, "buy-button")
                buy_button.click()
            elif side == "SELL":
                sell_button = order_form.find_element(By.CLASS_NAME, "sell-button")
                sell_button.click()
            
            # Wait for the order confirmation
            confirmation = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "order-confirmation"))
            )
            
            # Extract order details from the confirmation
            order_id = confirmation.find_element(By.CLASS_NAME, "order-id").text
            
            return {"order_id": order_id, "status": "PLACED"}
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None

    def get_positions(self):
        if not self.logged_in:
            self.login()
        
        try:
            # Navigate to the positions page
            self.driver.get("https://www.trading212.com/en/positions")
            
            # Wait for the positions table to be visible
            positions_table = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "positions-table"))
            )
            
            # Extract position information
            positions = []
            rows = positions_table.find_elements(By.TAG_NAME, "tr")
            for row in rows[1:]:  # Skip header row
                columns = row.find_elements(By.TAG_NAME, "td")
                position = {
                    "instrument": columns[0].text,
                    "quantity": columns[1].text,
                    "entry_price": columns[2].text,
                    "current_price": columns[3].text,
                    "profit_loss": columns[4].text
                }
                positions.append(position)
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return None

    def close(self):
        self.driver.quit()

# Note: This is a basic implementation and may need adjustments based on the actual Trading212 website structure