import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import logging

logger = logging.getLogger(__name__)

class AutoModelSelector:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'linear_regression': LinearRegression(),
            'elastic_net': ElasticNet(random_state=42),
            'svr': SVR(),
            'mlp': MLPRegressor(random_state=42),
            'xgboost': XGBRegressor(random_state=42),
            'lightgbm': LGBMRegressor(random_state=42)
        }
        self.best_model = None
        self.best_params = None

    def select_best_model(self, X, y):
        logger.info("Starting automatic model selection...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_score = float('inf')
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            score = self._optimize_model(model, X_train, y_train, X_test, y_test)
            if score < best_score:
                best_score = score
                self.best_model = model
                self.best_params = model.get_params()

        logger.info(f"Best model: {type(self.best_model).__name__}")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score (MSE): {best_score}")

        return self.best_model, self.best_params

    def _optimize_model(self, model, X_train, y_train, X_test, y_test):
        def objective(trial):
            params = self._get_model_params(trial, model)
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return mean_squared_error(y_test, y_pred)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        best_params = study.best_params
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Model: {type(model).__name__}, MSE: {mse}, R2: {r2}")
        return mse

    def _get_model_params(self, trial, model):
        if isinstance(model, RandomForestRegressor):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 10, 500),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
        elif isinstance(model, GradientBoostingRegressor):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 10, 500),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
        elif isinstance(model, ElasticNet):
            return {
                'alpha': trial.suggest_loguniform('alpha', 1e-5, 1.0),
                'l1_ratio': trial.suggest_uniform('l1_ratio', 0, 1)
            }
        elif isinstance(model, SVR):
            return {
                'C': trial.suggest_loguniform('C', 1e-5, 100),
                'epsilon': trial.suggest_loguniform('epsilon', 1e-5, 1.0),
                'gamma': trial.suggest_loguniform('gamma', 1e-5, 1.0)
            }
        elif isinstance(model, MLPRegressor):
            return {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)]),
                'alpha': trial.suggest_loguniform('alpha', 1e-5, 1.0),
                'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-5, 1.0)
            }
        elif isinstance(model, XGBRegressor):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 10, 500),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
            }
        elif isinstance(model, LGBMRegressor):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 10, 500),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0)
            }
        else:
            return {}

# You can add more model types and their respective hyperparameters here