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
    """Multi-Kernel Regression calculation."""
    weights = np.array([np.exp(-(i**2) / (2 * lookback**2)) for i in range(window + 1)])
    weights /= weights.sum()
    def apply_kernel(x): return np.dot(x[::-1], weights)
    return series.rolling(window=window+1).apply(apply_kernel, raw=True)

def fetch_and_prep_data(symbol):
    """Fetches data via Yahoo Finance to avoid Binance Geo-Blocks."""
    y_symbol = symbol.replace('/', '-').replace('USDT', 'USD')
    ticker = yf.Ticker(y_symbol)
    df = ticker.history(period="60d", interval="1h")
    
    if df.empty:
        raise ValueError(f"No data for {y_symbol}")

    # MKR Baseline
    df['mkr_line'] = calculate_mkr(df['Close'])
    
    # BBWP (Volatility Percentile)
    basis = df['Close'].rolling(20).mean()
    dev = 2.0 * df['Close'].rolling(20).std()
    bbw = (basis + dev - (basis - dev)) / basis
    df['bbwp'] = bbw.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    
    # PMARP (Momentum Percentile)
    df['pmar'] = df['Close'] / df['Close'].rolling(50).mean()
    df['pmarp'] = df['pmar'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    
    # MFI (Volume/Money Flow)
    df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    
    return df.dropna().iloc[-1]

# ==========================================
# 2. GA OPTIMIZER (FORCED DIVERSIFICATION)
# ==========================================
class MKROptimizer(ElementwiseProblem):
    def __init__(self, metrics):
        self.metrics = metrics
        self.n = len(metrics)
        # xu=0.3 strictly forces at least 4 coins to be chosen (100% / 30% = ~3.33)
        super().__init__(n_var=self.n, n_obj=1, n_ieq_constr=0, xl=np.zeros(self.n), xu=np.full(self.n, 0.3))

    def _evaluate(self, x, out, *args, **kwargs):
        # Normalize weights so they sum to 1.0
        s = np.sum(x)
        weights = x / s if s > 0 else np.ones(self.n) / self.n
        
        # Calculate Portfolio Metrics
        port_pf = np.dot(weights, [m['pf'] for m in self.metrics])
        port_calmar = np.dot(weights, [m['calmar'] for m in self.metrics])
        port_forecast = np.dot(weights, [m['forecast'] for m in self.metrics])
        
        # Fitness Score: 40% PF, 40% Calmar, 20% Forecast
        fitness = (0.4 * port_pf) + (0.4 * port_calmar) + (0.2 * port_forecast)
        
        # Pymoo minimizes, so we negate the fitness
        out["F"] = [-fitness]

# ==========================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    print("🚀 Running Diversified MKR Backend...")
    universe = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 
                'AVAX/USDT', 'NEAR/USDT', 'INJ/USDT', 'RNDR/USDT', 'STX/USDT']
    
    metrics = []
    for coin in universe:
        try:
            state = fetch_and_prep_data(coin)
            # STRATEGY FILTER: Price must be above MKR line
            is_bullish = state['Close'] > state['mkr_line']
            
            if is_bullish:
                # Calculate scores for optimization
                forecast = (state['pmarp'] / 100) * (state['mfi'] / 100) * 20 
                pf = 1.0 + (state['mfi'] / 50)
                calmar = max(0.1, 4.0 - (state['bbwp'] / 25))
                
                metrics.append({'Asset': coin, 'forecast': forecast, 'pf': pf, 'calmar': calmar})
                print(f"✅ BULLISH: {coin} (MFI: {int(state['mfi'])})")
            else:
                print(f"❌ BEARISH: {coin} (Skipped)")
        except Exception as e:
            print(f"⚠️ ERROR {coin}: {e}")

    if len(metrics) < 1:
        print("❌ CRITICAL: No bullish coins found. App will show empty.")
        final_data = []
    else:
        print(f"⚙️ Optimizing {len(metrics)} coins...")
        # Run GA Optimizer
        problem = MKROptimizer(metrics)
        algorithm = GA(pop_size=100)
        res = minimize(problem, algorithm, ('n_gen', 100), seed=42, verbose=False)
        
        # Extract and Normalize Weights
        optimal_weights = res.X / np.sum(res.X)
        
        final_data = []
        for i, m in enumerate(metrics):
            if optimal_weights[i] > 0.01: # Only include assets > 1% weight
                final_data.append({
                    "Asset": m['Asset'],
                    "Weight": round(float(optimal_weights[i]), 4),
                    "Forecast": round(float(m['forecast']), 2),
                    "Profit Factor": round(float(m['pf']), 2),
                    "Calmar": round(float(m['calmar']), 2)
                })

    # Wrap result with Timestamp for the iOS App
    output_package = {
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "portfolio": sorted(final_data, key=lambda x: x['Weight'], reverse=True)
    }

    # Write to data.json for Streamlit to read
    with open('data.json', 'w') as f:
        json.dump(output_package, f, indent=4)
    print(f"✅ SUCCESS: data.json updated with {len(final_data)} coins.")

    # Email notification (using GitHub Secrets)
    sender = os.environ.get('GMAIL_USER')
    pwd = os.environ.get('GMAIL_PASS')
    target = os.environ.get('MY_EMAIL')
    
    if sender and pwd and target:
        msg = MIMEMultipart()
        msg['Subject'] = f"🚀 MKR Portfolio Updated: {datetime.now().strftime('%H:%M')}"
        msg['From'] = f"MKR AI <{sender}>"
        msg['To'] = target
        body = f"Portfolio rebalanced. {len(final_data)} assets identified. Check your iOS app."
        msg.attach(MIMEText(body, 'plain'))
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                s.login(sender, pwd)
                s.send_message(msg)
            print("📧 Notification email sent.")
        except Exception as e:
            print(f"⚠️ Email failed: {e}")

if __name__ == "__main__":
    main()
