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
# 1. DUAL-SIDED STRATEGY MATH
# ==========================================
def calculate_mkr(series, lookback=12.0, window=30):
    weights = np.array([np.exp(-(i**2) / (2 * lookback**2)) for i in range(window + 1)])
    weights /= weights.sum()
    def apply_kernel(x): return np.dot(x[::-1], weights)
    return series.rolling(window=window+1).apply(apply_kernel, raw=True)

def fetch_and_analyze(symbol):
    y_symbol = symbol.replace('/', '-').replace('USDT', 'USD')
    df = yf.Ticker(y_symbol).history(period="60d", interval="1h")
    if df.empty: return None

    # Calculate Core Indicators
    df['mkr'] = calculate_mkr(df['Close'])
    df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    
    # Volatility Squeeze (BBWP)
    basis = df['Close'].rolling(20).mean()
    bbw = (2.0 * df['Close'].rolling(20).std() * 2) / basis
    df['bbwp'] = bbw.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    
    current = df.iloc[-1]
    
    # DUAL-SIDED LOGIC
    # Side 1: Long (Price > MKR)
    # Side -1: Short (Price < MKR)
    side = 1 if current['Close'] > current['mkr'] else -1
    
    # Forecast Score: Higher MFI is good for Longs, Lower MFI is good for Shorts
    mfi_score = current['mfi'] if side == 1 else (100 - current['mfi'])
    forecast = (mfi_score / 100) * 15 * side # Directional forecast
    
    # Risk Metrics
    pf = 1.0 + (mfi_score / 50)
    calmar = max(0.1, 4.0 - (current['bbwp'] / 25))
    
    return {
        "Asset": symbol,
        "Side": "LONG" if side == 1 else "SHORT",
        "forecast": abs(forecast), # Absolute potential
        "pf": pf,
        "calmar": calmar,
        "raw_side": side
    }

# ==========================================
# 2. OPTIMIZER (30% CAP)
# ==========================================
class DualSidedOptimizer(ElementwiseProblem):
    def __init__(self, metrics):
        self.metrics = metrics
        self.n = len(metrics)
        super().__init__(n_var=self.n, n_obj=1, n_ieq_constr=0, xl=np.zeros(self.n), xu=np.full(self.n, 0.3))

    def _evaluate(self, x, out, *args, **kwargs):
        s = np.sum(x)
        weights = x / s if s > 0 else np.ones(self.n) / self.n
        score = np.dot(weights, [m['pf'] * 0.4 + m['calmar'] * 0.4 + m['forecast'] * 0.2 for m in self.metrics])
        out["F"] = [-score]

# ==========================================
# 3. MAIN RUNNER
# ==========================================
def main():
    universe = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 'AVAX/USDT', 'NEAR/USDT', 'INJ/USDT', 'RNDR/USDT', 'STX/USDT']
    metrics = []
    
    for coin in universe:
        data = fetch_and_analyze(coin)
        if data: metrics.append(data)

    if metrics:
        res = minimize(DualSidedOptimizer(metrics), GA(pop_size=100), ('n_gen', 50), seed=42)
        optimal_weights = res.X / np.sum(res.X)
        
        portfolio = []
        for i, m in enumerate(metrics):
            if optimal_weights[i] > 0.02:
                portfolio.append({
                    "Asset": m['Asset'],
                    "Side": m['Side'],
                    "Weight": round(float(optimal_weights[i]), 4),
                    "Forecast": round(float(m['forecast']), 2),
                    "Profit Factor": round(float(m['pf']), 2),
                    "Calmar": round(float(m['calmar']), 2)
                })

        output = {
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "portfolio": sorted(portfolio, key=lambda x: x['Weight'], reverse=True)
        }
        with open('data.json', 'w') as f:
            json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()
