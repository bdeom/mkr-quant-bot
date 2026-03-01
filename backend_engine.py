import os
import json
import pandas as pd
import numpy as np
import ta
import yfinance as yf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

# ==========================================
# 1. CORE STRATEGY MATH
# ==========================================
def calculate_mkr(series, lookback=12.0, window=30):
    weights = np.array([np.exp(-(i**2) / (2 * lookback**2)) for i in range(window + 1)])
    weights /= weights.sum()
    def apply_kernel(x): return np.dot(x[::-1], weights)
    return series.rolling(window=window+1).apply(apply_kernel, raw=True)

def fetch_and_prep_data(symbol):
    # Map Binance symbols to Yahoo symbols
    y_symbol = symbol.replace('/', '-').replace('USDT', 'USD')
    ticker = yf.Ticker(y_symbol)
    df = ticker.history(period="60d", interval="1h")
    
    if df.empty:
        raise ValueError(f"No data for {y_symbol}")

    # MKR Baseline
    df['mkr_line'] = calculate_mkr(df['Close'])
    
    # BBWP (Volatility)
    basis = df['Close'].rolling(20).mean()
    dev = 2.0 * df['Close'].rolling(20).std()
    bbw = (basis + dev - (basis - dev)) / basis
    df['bbwp'] = bbw.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    
    # PMARP (Momentum)
    df['pmar'] = df['Close'] / df['Close'].rolling(50).mean()
    df['pmarp'] = df['pmar'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    
    # MFI (Volume)
    df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    
    return df.dropna().iloc[-1]

# ==========================================
# 2. OPTIMIZER (30% Diversification Cap)
# ==========================================
class MKROptimizer(ElementwiseProblem):
    def __init__(self, metrics):
        self.metrics = metrics
        self.n = len(metrics)
        # xu=0.3 forces the AI to never allocate more than 30% to one coin
        super().__init__(n_var=self.n, n_obj=1, n_ieq_constr=0, xl=np.zeros(self.n), xu=np.full(self.n, 0.3))

    def _evaluate(self, x, out, *args, **kwargs):
        if np.sum(x) == 0: x = np.ones(self.n) 
        weights = x / np.sum(x)
        
        port_pf = np.dot(weights, [m['pf'] for m in self.metrics])
        port_calmar = np.dot(weights, [m['calmar'] for m in self.metrics])
        port_forecast = np.dot(weights, [m['forecast'] for m in self.metrics])
        
        # Metric Protection Penalty
        penalty = 1000 if port_pf < 1.6 else 0
        
        # 40/40/20 Weighting Fitness
        fitness = (0.4 * port_pf) + (0.4 * port_calmar) + (0.2 * port_forecast)
        out["F"] = [-(fitness - penalty)]

# ==========================================
# 3. MAIN RUNNER
# ==========================================
def main():
    print("🚀 Running MKR Quant Backend...")
    universe = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 
                'AVAX/USDT', 'NEAR/USDT', 'INJ/USDT', 'RNDR/USDT', 'STX/USDT']
    
    metrics = []
    for coin in universe:
        try:
            state = fetch_and_prep_data(coin)
            is_bullish = state['Close'] > state['mkr_line']
            
            # Heuristics for GA ranking
            forecast = (state['pmarp'] / 100) * (state['mfi'] / 100) * (1 if is_bullish else -0.5) * 20 
            pf = 1.0 + (state['mfi'] / 50) if is_bullish else 0.8
            calmar = 4.0 - (state['bbwp'] / 25) 
            
            metrics.append({'Asset': coin, 'forecast': forecast, 'pf': pf, 'calmar': max(0.1, calmar)})
            print(f"✅ Data for {coin}")
        except Exception as e:
            print(f"⚠️ Skipping {coin}: {e}")

    if not metrics:
        return

    problem = MKROptimizer(metrics)
    res = minimize(problem, GA(pop_size=100), ('n_gen', 50), seed=42)
    
    optimal_weights = res.X / np.sum(res.X)
    
    portfolio = []
    for i, m in enumerate(metrics):
        if optimal_weights[i] > 0.01:
            portfolio.append({
                "Asset": m['Asset'],
                "Weight": round(optimal_weights[i], 4),
                "Forecast": round(m['forecast'], 2),
                "Profit Factor": round(m['pf'], 2),
                "Calmar": round(m['calmar'], 2)
            })
            
    # Wrap with Timestamp
    output = {
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "portfolio": sorted(portfolio, key=lambda x: x['Weight'], reverse=True)
    }

    with open('data.json', 'w') as f:
        json.dump(output, f, indent=4)
    print(f"✅ Success: data.json updated with {len(portfolio)} assets.")

    # Email notification
    sender = os.environ.get('GMAIL_USER')
    pwd = os.environ.get('GMAIL_PASS')
    target = os.environ.get('MY_EMAIL')
    if sender and pwd and target:
        msg = MIMEMultipart()
        msg['Subject'] = f"🚀 MKR Quant Updated: {datetime.now().strftime('%H:%M')}"
        msg.attach(MIMEText("Portfolio updated. Open iOS app to view.", 'plain'))
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                s.login(sender, pwd)
                s.send_message(msg)
        except: pass

if __name__ == "__main__":
    main()
