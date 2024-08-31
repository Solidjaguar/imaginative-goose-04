import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import mutual_info_score
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, window_size=1000, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_data = None

    def detect_drift(self, new_data):
        if self.reference_data is None:
            self.reference_data = new_data
            return False

        drift_detected = False
        for column in new_data.columns:
            if self._detect_univariate_drift(self.reference_data[column], new_data[column]):
                logger.warning(f"Drift detected in feature: {column}")
                drift_detected = True

        if self._detect_multivariate_drift(self.reference_data, new_data):
            logger.warning("Multivariate drift detected")
            drift_detected = True

        if drift_detected:
            self.reference_data = new_data  # Update reference data if drift is detected
        
        return drift_detected

    def _detect_univariate_drift(self, reference, new):
        statistic, p_value = ks_2samp(reference, new)
        return p_value < self.threshold

    def _detect_multivariate_drift(self, reference, new):
        ref_mi = self._compute_mutual_information(reference)
        new_mi = self._compute_mutual_information(new)
        mi_difference = np.abs(ref_mi - new_mi)
        return np.mean(mi_difference) > self.threshold

    def _compute_mutual_information(self, data):
        n_features = data.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i+1, n_features):
                mi = mutual_info_score(data.iloc[:, i], data.iloc[:, j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        return mi_matrix

# You can add more drift detection methods here as needed