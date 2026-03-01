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
# 1. FETCH & ENGINEER DATA (Via Yahoo Finance)
# ==========================================
def calculate_mkr(series, lookback=12.0, window=30):
    weights = np.array([np.exp(-(i**2) / (2 * lookback**2)) for i in range(window + 1)])
    weights /= weights.sum()
    def apply_kernel(x): return np.dot(x[::-1], weights)
    return series.rolling(window=window+1).apply(apply_kernel, raw=True)

def fetch_and_prep_data(symbol):
    # Convert Binance symbol (BTC/USDT) to Yahoo symbol (BTC-USD)
    y_symbol = symbol.replace('/', '-').replace('USDT', 'USD')
    
    # Fetch 4H equivalent (Yahoo provides 1h or 1d; we use 1h and resample)
    ticker = yf.Ticker(y_symbol)
    df = ticker.history(period="60d", interval="1h")
    
    if df.empty:
        raise ValueError(f"No data found for {y_symbol}")

    # Features
    df['mkr_line'] = calculate_mkr(df['Close'])
    
    # Trifecta
    basis = df['Close'].rolling(20).mean()
    dev = 2.0 * df['Close'].rolling(20).std()
    bbw = (basis + dev - (basis - dev)) / basis
    df['bbwp'] = bbw.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    
    df['pmar'] = df['Close'] / df['Close'].rolling(50).mean()
    df['pmarp'] = df['pmar'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    
    return df.dropna().iloc[-1]

# ==========================================
# 2. GA OPTIMIZER
# ==========================================
class MKROptimizer(ElementwiseProblem):
    def __init__(self, metrics):
        self.metrics = metrics
        self.n = len(metrics)
        super().__init__(n_var=self.n, n_obj=1, n_ieq_constr=0, xl=np.zeros(self.n), xu=np.ones(self.n))

    def _evaluate(self, x, out, *args, **kwargs):
        if np.sum(x) == 0: x = np.ones(self.n) 
        weights = x / np.sum(x)
        port_pf = np.dot(weights, [m['pf'] for m in self.metrics])
        port_calmar = np.dot(weights, [m['calmar'] for m in self.metrics])
        port_forecast = np.dot(weights, [m['forecast'] for m in self.metrics])
        penalty = 1000 if port_pf < 1.6 else 0
        fitness = (0.4 * port_pf) + (0.4 * port_calmar) + (0.2 * port_forecast)
        out["F"] = [-(fitness - penalty)]

# ==========================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    print("🚀 Starting MKR Quant Engine (Geo-Neutral Mode)...")
    universe = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 'AVAX/USDT', 'NEAR/USDT', 'INJ/USDT', 'RNDR/USDT', 'STX/USDT']
    
    metrics = []
    for coin in universe:
        try:
            state = fetch_and_prep_data(coin)
            is_bullish = state['Close'] > state['mkr_line']
            forecast = (state['pmarp'] / 100) * (state['mfi'] / 100) * (1 if is_bullish else -0.5) * 20 
            pf = 1.0 + (state['mfi'] / 50) if is_bullish else 0.8
            calmar = 4.0 - (state['bbwp'] / 25) 
            calmar = max(0.1, calmar)
            metrics.append({'Asset': coin, 'forecast': forecast, 'pf': pf, 'calmar': calmar})
            print(f"✅ Prepped {coin}")
        except Exception as e:
            print(f"⚠️ Skipping {coin}: {e}")

    if not metrics:
        print("❌ Error: No data fetched.")
        return

    print("Running Optimization...")
    problem = MKROptimizer(metrics)
    algorithm = GA(pop_size=100)
    res = minimize(problem, algorithm, ('n_gen', 50), seed=42, verbose=False)
    optimal_weights = res.X / np.sum(res.X)
    
    final_data = []
    for i, m in enumerate(metrics):
        if optimal_weights[i] > 0.01:
            final_data.append({
                "Asset": m['Asset'],
                "Weight": round(optimal_weights[i], 3),
                "Forecast": round(m['forecast'], 2),
                "Profit Factor": round(m['pf'], 2),
                "Calmar": round(m['calmar'], 2)
            })
            
    final_data = sorted(final_data, key=lambda x: x['Weight'], reverse=True)

    with open('data.json', 'w') as f:
        json.dump(final_data, f, indent=4)
    print(f"✅ Successfully wrote {len(final_data)} assets to data.json")

    # Email Logic (Unchanged)
    sender_email = os.environ.get('GMAIL_USER')
    sender_pass = os.environ.get('GMAIL_PASS')
    recipient_email = os.environ.get('MY_EMAIL')
    if sender_email and sender_pass and recipient_email:
        msg = MIMEMultipart()
        msg['Subject'] = f"🚀 MKR Rebalance Report: {datetime.now().strftime('%Y-%m-%d')}"
        msg['From'] = f"MKR AI <{sender_email}>"
        msg['To'] = recipient_email
        html = "<html><body><h2>Daily Portfolio Updated</h2><p>Your MKR iOS App has been updated.</p></body></html>"
        msg.attach(MIMEText(html, 'html'))
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_pass)
                server.send_message(msg)
            print("✅ Email sent.")
        except Exception as e: print(f"❌ Email failed: {e}")

if __name__ == "__main__":
    main()
