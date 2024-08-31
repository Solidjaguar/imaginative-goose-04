import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, window_size=30, threshold=2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_distribution = None
        self.baseline_performance = None

    def detect_drift(self, data):
        if self.baseline_distribution is None:
            self._set_baseline(data)
            return False

        recent_data = data.iloc[-self.window_size:]
        recent_distribution = self._calculate_distribution(recent_data)

        # Perform Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(self.baseline_distribution, recent_distribution)

        if p_value < 0.05 and ks_statistic > self.threshold:
            logger.warning(f"Drift detected: KS statistic = {ks_statistic}, p-value = {p_value}")
            return True
        
        return False

    def _set_baseline(self, data):
        self.baseline_distribution = self._calculate_distribution(data)
        logger.info("Baseline distribution set")

    def _calculate_distribution(self, data):
        return np.concatenate([data[col].values for col in data.columns if data[col].dtype in ['float64', 'int64']])

    def detect_performance_drift(self, model, X, y, threshold=0.1):
        if self.baseline_performance is None:
            self._set_baseline_performance(model, X, y)
            return False

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        current_performance = mean_squared_error(y_test, y_pred)

        performance_change = abs(current_performance - self.baseline_performance) / self.baseline_performance

        if performance_change > threshold:
            logger.warning(f"Performance drift detected: Change = {performance_change:.2f}")
            return True

        return False

    def _set_baseline_performance(self, model, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.baseline_performance = mean_squared_error(y_test, y_pred)
        logger.info(f"Baseline performance set: MSE = {self.baseline_performance:.4f}")

# You can add more drift detection methods here as needed