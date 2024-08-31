from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import optuna
import numpy as np
import pandas as pd
from src.backtesting.advanced_backtester import AdvancedBacktester
import logging

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    def __init__(self, model, strategy, risk_manager, sentiment_analyzer, economic_indicators):
        self.model = model
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.sentiment_analyzer = sentiment_analyzer
        self.economic_indicators = economic_indicators
        self.backtester = AdvancedBacktester(self.strategy, self.risk_manager, 
                                             self.sentiment_analyzer, self.economic_indicators)

    def optimize_price_predictor(self, X, y, param_distributions, n_iter=10, cv=5):
        scorer = make_scorer(lambda y_true, y_pred: -np.mean(np.abs(y_true - y_pred)))
        random_search = RandomizedSearchCV(self.model, param_distributions, n_iter=n_iter, 
                                           scoring=scorer, cv=cv, n_jobs=-1)
        random_search.fit(X, y)
        logger.info(f"Best parameters for price predictor: {random_search.best_params_}")
        logger.info(f"Best score for price predictor: {random_search.best_score_}")
        return random_search.best_estimator_

    def optimize_trading_strategy(self, data):
        def objective(trial):
            # Define the hyperparameters to optimize
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.0),
                'gamma': trial.suggest_uniform('gamma', 0.9, 0.9999),
                'n_steps': trial.suggest_int('n_steps', 16, 2048),
                'batch_size': trial.suggest_int('batch_size', 8, 256),
            }
            
            # Update the strategy with new parameters
            self.strategy.set_params(**params)
            
            # Run backtesting with the updated strategy
            backtest_results = self.backtester.run_backtest(data, data.index[0], data.index[-1])
            metrics = self.backtester.calculate_metrics()
            
            # Return the negative Sharpe ratio as we want to maximize it
            return -metrics['sharpe_ratio']

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        logger.info(f"Best parameters for trading strategy: {study.best_params}")
        logger.info(f"Best Sharpe ratio: {-study.best_value}")

        # Update the strategy with the best parameters
        self.strategy.set_params(**study.best_params)

    def optimize_risk_manager(self, data):
        def objective(trial):
            # Define the hyperparameters to optimize
            params = {
                'max_drawdown': trial.suggest_uniform('max_drawdown', 0.1, 0.5),
                'var_threshold': trial.suggest_uniform('var_threshold', 0.01, 0.1),
                'position_size': trial.suggest_uniform('position_size', 0.05, 0.3),
            }
            
            # Update the risk manager with new parameters
            self.risk_manager.set_params(**params)
            
            # Run backtesting with the updated risk manager
            backtest_results = self.backtester.run_backtest(data, data.index[0], data.index[-1])
            metrics = self.backtester.calculate_metrics()
            
            # Return a combined score (you can adjust this based on your priorities)
            return -metrics['sharpe_ratio'] + metrics['max_drawdown']

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        logger.info(f"Best parameters for risk manager: {study.best_params}")
        logger.info(f"Best combined score: {-study.best_value}")

        # Update the risk manager with the best parameters
        self.risk_manager.set_params(**study.best_params)

# You can add more optimization methods here as needed