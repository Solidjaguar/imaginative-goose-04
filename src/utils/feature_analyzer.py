import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    def __init__(self, importance_threshold=0.05):
        self.importance_threshold = importance_threshold
        self.feature_importance = None

    def analyze_features(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        mi_scores = mutual_info_regression(X_scaled, y)
        self.feature_importance = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        logger.info("Feature importance:")
        logger.info(self.feature_importance)
        
        return self.feature_importance

    def select_features(self, X):
        if self.feature_importance is None:
            logger.warning("Feature importance not calculated. Please run analyze_features first.")
            return X
        
        important_features = self.feature_importance[self.feature_importance > self.importance_threshold].index
        logger.info(f"Selected features: {important_features.tolist()}")
        return X[important_features]

    def engineer_features(self, X):
        # Example of simple feature engineering
        if 'Open' in X.columns and 'Close' in X.columns:
            X['Daily_Range'] = X['High'] - X['Low']
            X['Price_Change'] = X['Close'] - X['Open']
        
        if 'Volume' in X.columns:
            X['Log_Volume'] = np.log(X['Volume'] + 1)
        
        # Add more complex feature engineering here
        
        logger.info(f"Engineered features added: {[col for col in X.columns if col not in self.feature_importance.index]}")
        return X

# You can add more feature analysis and engineering methods here as needed