import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import logging

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'svr': SVR(kernel='rbf'),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        self.weights = None

    def train(self, X, y):
        logger.info("Training ensemble predictor...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        predictions = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions[name] = model.predict(X_val)

        # Calculate weights based on the inverse of MSE
        mse = {name: mean_squared_error(y_val, pred) for name, pred in predictions.items()}
        inv_mse = {name: 1/error for name, error in mse.items()}
        total_inv_mse = sum(inv_mse.values())
        self.weights = {name: value/total_inv_mse for name, value in inv_mse.items()}

        logger.info(f"Ensemble weights: {self.weights}")

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call 'train' before making predictions.")

        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        weighted_predictions = np.zeros(X.shape[0])
        for name, pred in predictions.items():
            weighted_predictions += self.weights[name] * pred

        return weighted_predictions

    def save_model(self, filename):
        joblib.dump(self, filename)

    @staticmethod
    def load_model(filename):
        return joblib.load(filename)

# You can add more ensemble methods or models here as needed