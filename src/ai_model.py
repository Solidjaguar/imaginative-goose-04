import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class GoldTradingAI:
    def __init__(self, lookback=30):
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_data(self, data):
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        scaled_data = self.scaler.fit_transform(data[features])
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(1 if scaled_data[i][3] > scaled_data[i-1][3] else 0)  # 1 if price went up, 0 if down
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    def train(self, data, epochs=50, batch_size=32):
        X, y = self.prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if self.model is None:
            self.build_model(X_train.shape[1:])
        
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
    def predict(self, data):
        X, _ = self.prepare_data(data.tail(self.lookback + 1))
        return self.model.predict(X)[-1][0]
    
    def generate_signals(self, data):
        signals = pd.Series(index=data.index, dtype='float64')
        for i in range(self.lookback, len(data)):
            prediction = self.predict(data.iloc[i-self.lookback:i+1])
            signals.iloc[i] = 1 if prediction > 0.5 else -1
        return signals

# You can add more AI model-related functions here as needed