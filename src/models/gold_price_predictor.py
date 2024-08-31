import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import dump, load

class GoldPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.best_model = None

    def prepare_data(self, data):
        X = data.drop(['target'], axis=1)
        y = data['target']
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train(self, data):
        X_train, X_test, y_train, y_test = self.prepare_data(data)

        models = {
            'RandomForest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor()
        }

        best_score = float('inf')
        for name, model in models.items():
            print(f"Training {name}...")
            param_grid = self.get_param_grid(name)
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            y_pred = grid_search.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            if mse < best_score:
                best_score = mse
                self.best_model = grid_search.best_estimator_
                print(f"New best model: {name}")
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"MSE: {mse}")
                print(f"R2 Score: {r2_score(y_test, y_pred)}")

        self.model = self.best_model

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

    def save_model(self, filepath):
        dump(self.model, filepath)

    def load_model(self, filepath):
        self.model = load(filepath)

# You can add more model-related functions here as needed