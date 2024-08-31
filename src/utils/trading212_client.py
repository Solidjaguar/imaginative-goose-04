import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Trading212Client:
    def __init__(self, api_key: str, account_id: str):
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = "https://api.trading212.com/v1"  # Replace with actual API URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_account_info(self) -> Dict[str, Any]:
        """Fetch account information including balance"""
        endpoint = f"{self.base_url}/accounts/{self.account_id}"
        response = self._make_request("GET", endpoint)
        return response.json()

    def place_order(self, instrument: str, quantity: float, side: str, type: str, limit_price: float = None) -> Dict[str, Any]:
        """Place a new order"""
        endpoint = f"{self.base_url}/orders"
        payload = {
            "accountId": self.account_id,
            "instrument": instrument,
            "quantity": quantity,
            "side": side,
            "type": type
        }
        if limit_price:
            payload["limitPrice"] = limit_price
        response = self._make_request("POST", endpoint, json=payload)
        return response.json()

    def get_positions(self) -> Dict[str, Any]:
        """Fetch current positions"""
        endpoint = f"{self.base_url}/accounts/{self.account_id}/positions"
        response = self._make_request("GET", endpoint)
        return response.json()

    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an HTTP request to the Trading212 API"""
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to Trading212 API: {e}")
            raise

# Note: This is a mock implementation. Replace with actual Trading212 API when available.