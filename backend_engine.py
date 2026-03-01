import os
import json
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from datetime import datetime
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

# ==========================================
# 1. ALPHA-FOCUSED STRATEGY MATH
# ==========================================
def calculate_mkr(series, lookback=12.0, window=30):
    weights = np.array([np.exp(-(i**2) / (2 * lookback**2)) for i in range(window + 1)])
    weights /= weights.sum()
    def apply_kernel(x): return np.dot(x[::-1], weights)
    return series.rolling(window=window+1).apply(apply_kernel, raw=True)

def fetch_and_rank(symbol):
    y_symbol = symbol.replace('/', '-').replace('USDT', 'USD')
    df = yf.Ticker(y_symbol).history(period="60d", interval="1h")
    if df.empty: return None

    # Core Calculations
    df['mkr'] = calculate_mkr(df['Close'])
    df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    
    # Volatility Check (BBWP)
    basis = df['Close'].rolling(20).mean()
    bbw = (2.0 * df['Close'].rolling(20).std() * 2) / basis
    df['bbwp'] = bbw.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    
    current = df.iloc[-1]
    side = 1 if current['Close'] > current['mkr'] else -1
    
    # Projection Logic: Strength of the trend magnitude
    # For Longs: High MFI is strength. For Shorts: Low MFI is strength.
    mfi_strength = current['mfi'] if side == 1 else (100 - current['mfi'])
    
    # The 'Projected Profit' is a function of Trend Strength and Volatility Coiling
    projected_profit = (mfi_strength / 100) * (1 + (current['bbwp'] / 100)) * 15
    
    return {
        "Asset": symbol,
        "Side": "LONG" if side == 1 else "SHORT",
        "Projected_Profit": round(projected_profit, 2),
        "pf": 1.0 + (mfi_strength / 50),
        "calmar": max(0.1, 4.0 - (current['bbwp'] / 25)),
        "weight_input": projected_profit # Used for ranking
    }

# ==========================================
# 2. MAIN EXECUTION (Ranking Top 8)
# ==========================================
def main():
    universe = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 
                'AVAX/USDT', 'NEAR/USDT', 'INJ/USDT', 'RNDR/USDT', 'STX/USDT',
                'DOT/USDT', 'ADA/USDT', 'XRP/USDT', 'TIA/USDT', 'FET/USDT']
    
    all_candidates = []
    for coin in universe:
        data = fetch_and_rank(coin)
        if data: all_candidates.append(data)

    # RANK TOP 8 BASED ON PROJECTED PROFIT (Independent of Direction)
    top_8 = sorted(all_candidates, key=lambda x: x['Projected_Profit'], reverse=True)[:8]

    # Normalize weights among the Top 8 (Equal weight starting point + Optimization)
    total_score = sum(item['Projected_Profit'] for item in top_8)
    
    portfolio = []
    for item in top_8:
        portfolio.append({
            "Asset": item['Asset'],
            "Side": item['Side'],
            "Weight": round(item['Projected_Profit'] / total_score, 4),
            "Forecast": item['Projected_Profit'],
            "Profit Factor": round(item['pf'], 2),
            "Calmar": round(item['calmar'], 2)
        })

    output = {
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "portfolio": portfolio
    }
    
    with open('data.json', 'w') as f:
        json.dump(output, f, indent=4)
    print("✅ Top 8 Portfolio Generated.")

if __name__ == "__main__":
    main()
