import numpy as np
import pandas as pd
from Modules.enums import FNO, Option

class DataForIndicators:
    def __init__(self, df, col):
        self.df = df
        self.col = col

    def rsi(self, window=2):
        delta = self.df[self.col].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        return rsi
    
    def bollinger_band(self, std=2, window=275):
        ma = self.df[self.col].rolling(window=window).mean()
        std = self.df[self.col].rolling(window=window).std()
        upperbb = ma + 2 * std
        lowerbb = ma - 2 * std
        return upperbb, lowerbb
    

# ic['4*std 50'] = 4 * std
# ic['Zscore 1500'] = (ic['ic'] - ic['ic'].rolling(window=1500).mean())/(ic['ic'].rolling(window=1500).std())
# ic['Zscore 750'] = (ic['ic'] - ic['ic'].rolling(window=750).mean())/(ic['ic'].rolling(window=750).std())
# ic['Zscore 375'] = (ic['ic'] - ic['ic'].rolling(window=375).mean())/(ic['ic'].rolling(window=375).std())
# ic['Zscore 100'] = (ic['ic'] - ic['ic'].rolling(window=100).mean())/(ic['ic'].rolling(window=100).std())
# ic['Zscore 4*std'] = (4*std - (4*std).rolling(window=375).mean())/((4*std).rolling(window=375).std())
# ic['Zscore 4*std 50'] = (4*std - (4*std).rolling(window=50).mean())/((4*std).rolling(window=50).std())
# # rsi = calculate_rsi(ic['ic'], 2)
# # ic['2 period RSI Upper'] = rsi[rsi > 0.9]
# # ic['2 period RSI Lower'] = rsi[rsi < 0.1]

# ic['IC EMA Fast'] = ic['ic'].ewm(span=15).mean() 
# ic['IC EMA Slow'] = ic['ic'].ewm(span=100).mean() 
# ic['IC EMA Difference'] = ic['IC EMA Fast'] - ic['IC EMA Slow']
# ic['EMA Fast of EMA Difference'] = ic['IC EMA Difference'].ewm(span=10).mean()
# ic['EMA Slow of EMA Difference'] = ic['IC EMA Difference'].ewm(span=40).mean()

# # ic['Zscore EMA Fast'] = ic['Zscore 750'].ewm(span=15).mean() 
# # ic['Zscore EMA Slow'] = ic['Zscore 750'].ewm(span=100).mean() 
# # ic['Zscore EMA Difference'] = ic['Zscore EMA Fast'] - ic['Zscore EMA Slow']

# ic['Smooth IC'] = ic['ic'].ewm(span=6).mean()
# ic['S Smooth IC'] = ic['Smooth IC'].ewm(span=3).mean()
# ic['S S Smooth IC'] = ic['S Smooth IC'].ewm(span=3).mean()
# ic['S S S Smooth IC'] = ic['S S Smooth IC'].ewm(span=3).mean()