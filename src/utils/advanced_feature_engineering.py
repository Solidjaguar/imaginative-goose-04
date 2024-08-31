import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import talib
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% of variance

    def engineer_features(self, data):
        logger.info("Starting advanced feature engineering...")
        
        # Add all technical indicators
        data = add_all_ta_features(
            data, open="Open", high="High", low="Low", close="Close", volume="Volume"
        )
        
        # Add custom features
        data = self.add_custom_features(data)
        
        # Add cyclical features for time-based columns
        data = self.add_cyclical_features(data)
        
        # Add rolling statistics
        data = self.add_rolling_features(data)
        
        # Add lag features
        data = self.add_lag_features(data)
        
        # Add interaction features
        data = self.add_interaction_features(data)
        
        # Perform feature scaling
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = self.scaler.fit_transform(data[numeric_columns])
        
        # Perform PCA
        pca_result = self.pca.fit_transform(data[numeric_columns])
        pca_df = pd.DataFrame(
            pca_result, 
            columns=[f'PC_{i+1}' for i in range(pca_result.shape[1])], 
            index=data.index
        )
        data = pd.concat([data, pca_df], axis=1)
        
        logger.info("Advanced feature engineering completed.")
        return data

    def add_custom_features(self, data):
        # Gold to S&P500 ratio
        data['Gold_SP500_Ratio'] = data['Close'] / data['SP500']
        
        # Gold to USD/EUR ratio
        data['Gold_USDEUR_Ratio'] = data['Close'] / data['USDEUR']
        
        # Relative strength index
        data['RSI'] = talib.RSI(data['Close'])
        
        # Bollinger Bands
        data['Upper_BB'], data['Middle_BB'], data['Lower_BB'] = talib.BBANDS(data['Close'])
        
        # MACD
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['Close'])
        
        return data

    def add_cyclical_features(self, data):
        data['Day_Sin'] = np.sin(2 * np.pi * data.index.dayofyear / 365.25)
        data['Day_Cos'] = np.cos(2 * np.pi * data.index.dayofyear / 365.25)
        data['Month_Sin'] = np.sin(2 * np.pi * data.index.month / 12)
        data['Month_Cos'] = np.cos(2 * np.pi * data.index.month / 12)
        return data

    def add_rolling_features(self, data):
        windows = [7, 14, 30, 90]
        for window in windows:
            data[f'Rolling_Mean_{window}d'] = data['Close'].rolling(window=window).mean()
            data[f'Rolling_Std_{window}d'] = data['Close'].rolling(window=window).std()
            data[f'Rolling_Min_{window}d'] = data['Close'].rolling(window=window).min()
            data[f'Rolling_Max_{window}d'] = data['Close'].rolling(window=window).max()
        return data

    def add_lag_features(self, data):
        lags = [1, 3, 7, 14, 30]
        for lag in lags:
            data[f'Lag_{lag}d'] = data['Close'].shift(lag)
        return data

    def add_interaction_features(self, data):
        data['Price_Volume_Interaction'] = data['Close'] * data['Volume']
        data['RSI_Volume_Interaction'] = data['RSI'] * data['Volume']
        return data

# You can add more feature engineering methods here as needed