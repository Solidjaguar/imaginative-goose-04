import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import logging

logger = logging.getLogger(__name__)

class PatternDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.hmm = hmm.GaussianHMM(n_components=3, covariance_type="full", random_state=42)
        self.scaler = StandardScaler()

    def detect_anomalies(self, data):
        logger.info("Detecting anomalies using Isolation Forest...")
        scaled_data = self.scaler.fit_transform(data)
        anomalies = self.isolation_forest.fit_predict(scaled_data)
        return pd.Series(anomalies, index=data.index, name='Anomaly')

    def detect_regimes(self, data):
        logger.info("Detecting market regimes using K-means clustering...")
        scaled_data = self.scaler.fit_transform(data)
        regimes = self.kmeans.fit_predict(scaled_data)
        return pd.Series(regimes, index=data.index, name='Regime')

    def detect_hidden_states(self, data):
        logger.info("Detecting hidden states using Hidden Markov Model...")
        scaled_data = self.scaler.fit_transform(data)
        self.hmm.fit(scaled_data)
        hidden_states = self.hmm.predict(scaled_data)
        return pd.Series(hidden_states, index=data.index, name='HiddenState')

    def detect_patterns(self, data):
        logger.info("Starting pattern detection...")
        
        # Select relevant features for pattern detection
        features_for_patterns = [
            'Close', 'Volume', 'RSI', 'MACD', 'Upper_BB', 'Lower_BB',
            'Gold_SP500_Ratio', 'Gold_USDEUR_Ratio'
        ]
        pattern_data = data[features_for_patterns]
        
        # Detect anomalies
        anomalies = self.detect_anomalies(pattern_data)
        
        # Detect market regimes
        regimes = self.detect_regimes(pattern_data)
        
        # Detect hidden states
        hidden_states = self.detect_hidden_states(pattern_data)
        
        # Combine all pattern detections
        patterns = pd.concat([anomalies, regimes, hidden_states], axis=1)
        
        logger.info("Pattern detection completed.")
        return patterns

    def generate_trading_signals(self, patterns, risk_tolerance='low'):
        logger.info("Generating trading signals based on detected patterns...")
        
        signals = pd.Series(index=patterns.index, dtype='float')
        
        # Define risk levels
        risk_levels = {
            'low': {'anomaly_threshold': -0.8, 'regime_threshold': 2},
            'medium': {'anomaly_threshold': -0.5, 'regime_threshold': 1},
            'high': {'anomaly_threshold': -0.2, 'regime_threshold': 0}
        }
        
        threshold = risk_levels.get(risk_tolerance, risk_levels['low'])
        
        for i in range(len(patterns)):
            if patterns['Anomaly'].iloc[i] == -1 and patterns['Anomaly'].iloc[i] <= threshold['anomaly_threshold']:
                # Strong anomaly detected, potential buying opportunity
                signals.iloc[i] = 1
            elif patterns['Regime'].iloc[i] >= threshold['regime_threshold']:
                # Favorable market regime, potential buying opportunity
                signals.iloc[i] = 1
            elif patterns['HiddenState'].iloc[i] == 2:  # Assuming state 2 is the most bullish state
                # Bullish hidden state detected, potential buying opportunity
                signals.iloc[i] = 1
            elif patterns['Anomaly'].iloc[i] == 1 or patterns['Regime'].iloc[i] == 0:
                # Normal market conditions or unfavorable regime, hold
                signals.iloc[i] = 0
            else:
                # Potentially bearish conditions, sell
                signals.iloc[i] = -1
        
        logger.info("Trading signals generated.")
        return signals

# You can add more pattern detection methods here as needed