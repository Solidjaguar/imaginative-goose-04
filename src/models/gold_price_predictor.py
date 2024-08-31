import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import dump, load
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldPricePredictor:
    def __init__(self, retrain_interval=30, performance_threshold=0.1):
        self.model = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.retrain_interval = retrain_interval  # Days between retraining
        self.performance_threshold = performance_threshold  # Threshold for model degradation
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

        models = {
            'RandomForest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor()
        }

        best_score = float('inf')
        for name, model in models.items():
            logger.info(f"Training {name}...")
            param_grid = self.get_param_grid(name)
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            y_pred = grid_search.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            if mse < best_score:
                best_score = mse
                self.best_model = grid_search.best_estimator_
                logger.info(f"New best model: {name}")
                logger.info(f"Best parameters: {grid_search.best_params_}")
                logger.info(f"MSE: {mse}")
                logger.info(f"R2 Score: {r2_score(y_test, y_pred)}")

        self.model = self.best_model
        self.last_train_date = current_date
        self.baseline_performance = best_score
        logger.info(f"Model trained. Baseline performance (MSE): {self.baseline_performance}")

    def get_param_grid(self, model_name):
        if model_name == 'RandomForest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'XGBoost':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        elif model_name == 'LightGBM':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [-1, 10, 20, 30],
                'learning_rate': [0.01, 0.1, 0.3]
            }

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
        
        # Incremental learning for supported models
        if isinstance(self.model, (XGBRegressor, LGBMRegressor)):
            X, y = self.prepare_data(new_data)
            self.model.fit(X, y, xgb_model=self.model if isinstance(self.model, XGBRegressor) else None)
            logger.info("Model updated incrementally.")
        else:
            logger.info("Current model doesn't support incremental learning. Consider retraining if performance degrades.")

    def save_model(self, filepath):
        dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = load(filepath)
        logger.info(f"Model loaded from {filepath}")

# You can add more model-related functions here as needed