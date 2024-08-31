import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import dump, load
import logging

logger = logging.getLogger(__name__)

class EnsembleModel:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor()
        }
        self.weights = {model: 1/len(self.models) for model in self.models}

    def fit(self, X, y):
        for name, model in self.models.items():
            model.fit(X, y)
            logger.info(f"Trained {name} model")

    def predict(self, X):
        predictions = np.zeros(len(X))
        for name, model in self.models.items():
            predictions += self.weights[name] * model.predict(X)
        return predictions

    def update_weights(self, X, y):
        performances = {}
        for name, model in self.models.items():
            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)
            performances[name] = 1 / mse  # Use inverse MSE as weight

        total_performance = sum(performances.values())
        self.weights = {name: perf / total_performance for name, perf in performances.items()}
        logger.info(f"Updated model weights: {self.weights}")

class GoldPricePredictor:
    def __init__(self, retrain_interval=30, performance_threshold=0.1):
        self.model = EnsembleModel()
        self.scaler = StandardScaler()
        self.retrain_interval = retrain_interval
        self.performance_threshold = performance_threshold
        self.last_train_date = None
        self.baseline_performance = None

    def prepare_data(self, data):
        X = data.drop(['target'], axis=1)
        y = data['target']
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train(self, data, force=False):
        current_date = pd.Timestamp.now()
        
        if not force and self.last_train_date and (current_date - self.last_train_date).days < self.retrain_interval:
            logger.info("Skipping training: Not enough time has passed since last training.")
            return

        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Ensemble Model - MSE: {mse}, R2 Score: {r2}")

        self.last_train_date = current_date
        self.baseline_performance = mse
        logger.info(f"Model trained. Baseline performance (MSE): {self.baseline_performance}")

    def predict(self, data):
        X = self.scaler.transform(data)
        return self.model.predict(X)

    def evaluate_performance(self, data):
        X, y = self.prepare_data(data)
        y_pred = self.predict(X)
        current_performance = mean_squared_error(y, y_pred)
        logger.info(f"Current performance (MSE): {current_performance}")
        
        if self.baseline_performance:
            performance_change = (current_performance - self.baseline_performance) / self.baseline_performance
            logger.info(f"Performance change: {performance_change:.2%}")
            
            if performance_change > self.performance_threshold:
                logger.warning("Performance degradation detected. Initiating retraining.")
                self.train(data, force=True)
            else:
                logger.info("Model performance is within acceptable range.")
        else:
            logger.warning("No baseline performance available. Unable to evaluate performance change.")

    def update(self, new_data):
        logger.info("Updating model with new data...")
        self.evaluate_performance(new_data)
        
        X, y = self.prepare_data(new_data)
        self.model.update_weights(X, y)

    def save_model(self, filepath):
        dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = load(filepath)
        logger.info(f"Model loaded from {filepath}")

# You can add more model-related functions here as needed