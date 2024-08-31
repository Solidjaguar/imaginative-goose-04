import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LiquidityEstimator:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, data):
        features = pd.DataFrame({
            'volume': data['volume'],
            'price_range': data['high'] - data['low'],
            'returns': data['close'].pct_change(),
            'volatility': data['close'].pct_change().rolling(window=20).std(),
            'bid_ask_spread': data['ask'] - data['bid'],  # Assuming we have bid-ask data
            'depth_imbalance': (data['bid_size'] - data['ask_size']) / (data['bid_size'] + data['ask_size']),
            'trade_size': data['volume'] / data['trades_count'],  # Assuming we have trades count data
            'time_of_day': pd.to_datetime(data.index).hour + pd.to_datetime(data.index).minute / 60,
            'is_weekend': pd.to_datetime(data.index).dayofweek.isin([5, 6]).astype(int),
        })
        return features.dropna()

    def train(self, data):
        features = self.prepare_features(data)
        target = data['volume'] / data['price_range']  # Simple liquidity measure

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        score = self.model.score(X_test_scaled, y_test)
        print(f"Liquidity Estimator R^2 Score: {score}")

    def estimate_liquidity(self, current_data):
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() method first.")

        features = self.prepare_features(current_data.tail(1))
        features_scaled = self.scaler.transform(features)
        liquidity_estimate = self.model.predict(features_scaled)[0]

        return liquidity_estimate

    def get_feature_importance(self):
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() method first.")

        feature_importance = pd.DataFrame({
            'feature': self.prepare_features(pd.DataFrame()).columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importance

# You can add more liquidity estimation methods here as needed