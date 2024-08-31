import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ExecutionQualityPredictor:
    def __init__(self):
        self.slippage_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.execution_time_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, data):
        features = pd.DataFrame({
            'volume': data['volume'],
            'volatility': data['close'].pct_change().rolling(window=20).std(),
            'bid_ask_spread': data['ask'] - data['bid'],
            'order_size': data['order_size'],
            'market_impact': data['order_size'] / data['volume'],
            'time_of_day': pd.to_datetime(data.index).hour + pd.to_datetime(data.index).minute / 60,
            'is_weekend': pd.to_datetime(data.index).dayofweek.isin([5, 6]).astype(int),
            'liquidity_estimate': data['liquidity_estimate'],
        })
        return features.dropna()

    def train(self, data):
        features = self.prepare_features(data)
        slippage_target = data['slippage']
        execution_time_target = data['execution_time']

        X_train, X_test, y_slippage_train, y_slippage_test, y_time_train, y_time_test = train_test_split(
            features, slippage_target, execution_time_target, test_size=0.2, random_state=42)

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.slippage_model.fit(X_train_scaled, y_slippage_train)
        self.execution_time_model.fit(X_train_scaled, y_time_train)
        self.is_trained = True

        slippage_score = self.slippage_model.score(X_test_scaled, y_slippage_test)
        execution_time_score = self.execution_time_model.score(X_test_scaled, y_time_test)
        print(f"Slippage Predictor R^2 Score: {slippage_score}")
        print(f"Execution Time Predictor R^2 Score: {execution_time_score}")

    def predict_execution_quality(self, current_data):
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() method first.")

        features = self.prepare_features(current_data.tail(1))
        features_scaled = self.scaler.transform(features)

        predicted_slippage = self.slippage_model.predict(features_scaled)[0]
        predicted_execution_time = self.execution_time_model.predict(features_scaled)[0]

        return {
            'predicted_slippage': predicted_slippage,
            'predicted_execution_time': predicted_execution_time
        }

    def get_feature_importance(self):
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() method first.")

        slippage_importance = pd.DataFrame({
            'feature': self.prepare_features(pd.DataFrame()).columns,
            'slippage_importance': self.slippage_model.feature_importances_
        }).sort_values('slippage_importance', ascending=False)

        execution_time_importance = pd.DataFrame({
            'feature': self.prepare_features(pd.DataFrame()).columns,
            'execution_time_importance': self.execution_time_model.feature_importances_
        }).sort_values('execution_time_importance', ascending=False)

        return slippage_importance, execution_time_importance

# You can add more execution quality prediction methods here as needed