"""
PyQuant Library - Vectorized Technical Indicators
Script: Intraday VWAP (with Daily Session Reset) & Vectorized Supertrend
Full Library & Live Sandbox: https://pyquant-library.netlify.app
"""

import pandas as pd
import numpy as np


def calculate_intraday_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Intraday Volume Weighted Average Price (VWAP) with an automatic
    daily session reset. Expects a DataFrame with a DatetimeIndex and
    'High', 'Low', 'Close', 'Volume' columns.
    """
    df = df.copy()
    
    # Calculate Typical Price and Volume-weighted Typical Price
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    tp_volume = typical_price * df['Volume']
    
    # Group by trading session date to reset cumulative sums each morning
    session_date = df.index.date
    cum_tp_vol = tp_volume.groupby(session_date).cumsum()
    cum_vol = df['Volume'].groupby(session_date).cumsum()
    
    df['VWAP'] = cum_tp_vol / np.where(cum_vol == 0, np.nan, cum_vol)
    return df


def calculate_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Calculates the Supertrend indicator using vectorized True Range / ATR
    and fast NumPy array traversal for band locking.
    """
    df = df.copy()
    
    # 1. Vectorized True Range (TR) & Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    
    # 2. Basic Upper and Lower Bands
    hl2 = (df['High'] + df['Low']) / 2.0
    upper_band = (hl2 + (multiplier * atr)).values
    lower_band = (hl2 - (multiplier * atr)).values
    close = df['Close'].values
    
    # 3. Fast NumPy array traversal for stateful band locking
    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n, dtype=int)  # 1 = Bullish, -1 = Bearish
    
    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    supertrend[0] = lower_band[0]
    
    for i in range(1, n):
        # Lock Upper Band
        if upper_band[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i - 1]
            
        # Lock Lower Band
        if lower_band[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i - 1]
            
        # Determine Trend Direction
        if direction[i - 1] == -1 and close[i] > final_upper[i]:
            direction[i] = 1
        elif direction[i - 1] == 1 and close[i] < final_lower[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            
        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
        
    df['Supertrend'] = np.round(supertrend, 2)
    df['Trend_Direction'] = direction
    return df


if __name__ == "__main__":
    # Quick standalone test with synthetic 5-minute intraday OHLCV data
    np.random.seed(42)
    dates = pd.date_range("2026-11-02 09:15", periods=50, freq="5min")
    base_price = 24000 + np.cumsum(np.random.randn(50) * 15)
    
    sample_df = pd.DataFrame({
        "Open": base_price,
        "High": base_price + np.random.uniform(5, 25, size=50),
        "Low": base_price - np.random.uniform(5, 25, size=50),
        "Close": base_price + np.random.uniform(-10, 10, size=50),
        "Volume": np.random.randint(1000, 25000, size=50)
    }, index=dates)
    
    sample_df = calculate_intraday_vwap(sample_df)
    sample_df = calculate_supertrend(sample_df, period=10, multiplier=3.0)
    
    print("=== PyQuant Vectorized VWAP & Supertrend Sample Output ===")
    print(sample_df[['Close', 'VWAP', 'Supertrend', 'Trend_Direction']].tail(10))
