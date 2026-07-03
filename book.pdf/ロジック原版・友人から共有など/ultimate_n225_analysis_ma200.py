import yfinance as yf
import pandas as pd
import urllib.request
import re
import os
import time
from datetime import datetime

def get_nikkei_225_tickers():
    """Read Nikkei 225 tickers from a local manual list file."""
    path = "n225_manual_list.txt"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return []
    try:
        with open(path, 'r') as f:
            content = f.read()
            # Split by comma or newline and clean
            codes = [c.strip() for c in re.split(r'[,\n]', content) if c.strip()]
            unique = []
            for c in codes:
                if c not in unique:
                    unique.append(c)
            print(f"Loaded {len(unique)} tickers from manual list.")
            return unique
    except Exception as e:
        print(f"File Read Error: {e}")
        return []

def calc_rma(series, length):
    rma = pd.Series(index=series.index, dtype=float)
    if len(series) < length: return rma
    rma.iloc[length-1] = series.iloc[:length].mean()
    alpha = 1.0 / length
    for i in range(length, len(series)):
        rma.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * rma.iloc[i-1]
    return rma

def calc_sma(series, length):
    return series.rolling(window=length).mean()

def rsi_pine(series, length=10):
    delta = series.diff()
    u = delta.where(delta > 0, 0.0)
    d = -delta.where(delta < 0, 0.0)
    rs_up = calc_rma(u, length)
    rs_down = calc_rma(d, length)
    rsi = 100.0 - (100.0 / (1.0 + rs_up / rs_down))
    rsi[rs_down == 0] = 100.0
    return rsi

def get_robust_monthly_data(ticker):
    """Try monthly, fallback to weekly resampled if needed. Also fetch daily for SMA200."""
    try:
        # Fetch Daily first for SMA200 (need enough history for 200-day SMA)
        df_d = yf.download(ticker, interval="1d", period="max", progress=False)
        if isinstance(df_d.columns, pd.MultiIndex):
            df_d.columns = [c[0] for c in df_d.columns]
        
        sma200_d = pd.Series(dtype=float)
        if not df_d.empty and len(df_d) >= 200:
            sma200_d = calc_sma(df_d['Close'], 200)
            
        # Try direct monthly
        df = yf.download(ticker, interval="1mo", period="max", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
            
        if df.empty or len(df) < 20:
            # Fallback to weekly resampled to monthly
            print(f"Monthly incomplete for {ticker}, trying weekly fallback...")
            df_wk = yf.download(ticker, interval="1wk", period="max", progress=False)
            if isinstance(df_wk.columns, pd.MultiIndex):
                df_wk.columns = [c[0] for c in df_wk.columns]
            if not df_wk.empty:
                df = df_wk.resample('ME').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        
        # Merge SMA200 from daily (take last available SMA200 for the month)
        if not sma200_d.empty:
            sma200_m = sma200_d.resample('ME').last()
            # Align indices: Convert both to period 'M' to merge on Year-Month
            df.index = df.index.to_period('M')
            sma200_m.index = sma200_m.index.to_period('M')
            df = df.join(sma200_m.rename('SMA200'), how='left')
            # Convert back to timestamp (defaults to first day of month)
            df.index = df.index.to_timestamp()
        else:
            df['SMA200'] = None
            
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
    return pd.DataFrame()

def check_profit_with_daily(ticker, entry_price, start_date):
    """Download daily data to check for any high > entry_price."""
    try:
        # Download from entry month start to now
        df_d = yf.download(ticker, start=start_date, interval="1d", progress=False)
        if df_d.empty:
            return False
        max_daily_high = df_d['High'].max()
        # User defined: "+ 1 yen" or just "strictly greater"
        # We'll use > entry_price as the condition
        return max_daily_high > entry_price
    except:
        return False

def main():
    tickers = get_nikkei_225_tickers()
    if not tickers:
        print("COULD_NOT_FETCH_TICKERS")
        return

    print(f"Processing {len(tickers)} tickers with Multi-Timeframe Validation (MA200 Filtered)...")
    
    all_signals = []
    
    for idx, ticker in enumerate(tickers):
        print(f"[{idx+1}/{len(tickers)}] Segmenting {ticker}...")
        df = get_robust_monthly_data(ticker)
        
        if df.empty or len(df) < 15:
            print(f"Skipping {ticker}: Insufficient data.")
            continue
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
            
        df = df.dropna(subset=['Close'])
        df['RSI'] = rsi_pine(df['Close'], 10)
        df['RSI_SMA'] = df['RSI'].rolling(3).mean()
        df['Signal'] = (df['RSI'] > df['RSI_SMA']) & (df['RSI'].shift(1) <= df['RSI_SMA'].shift(1))
        
        armed = False
        for i in range(1, len(df)):
            try:
                # Explicit scalar conversion/check for RSI
                prev_rsi = float(df['RSI'].iloc[i-1])
                if prev_rsi < 50:
                    armed = True
                
                # Check for Signal (Crossover)
                signal_val = bool(df['Signal'].iloc[i])
                
                if armed and signal_val:
                    armed = False
                    if i + 1 < len(df):
                        signal_date = df.index[i]
                        entry_date = df.index[i+1]
                        entry_price = float(df.iloc[i+1]['Open'])
                        
                        # MA200 Screen check
                        current_close = float(df['Close'].iloc[i])
                        sma200_val = df['SMA200'].iloc[i]
                        passed_ma200 = False
                        if pd.notnull(sma200_val):
                            passed_ma200 = current_close > float(sma200_val)
                        
                        # Only proceed if MA200 criteria is met
                        if not passed_ma200:
                            armed = True
                            continue

                        # Monthly check
                        try:
                            prices_slice = df.iloc[i+1:]['High']
                            if not prices_slice.empty:
                                max_month_high = float(prices_slice.max())
                            else:
                                max_month_high = 0.0
                        except:
                            max_month_high = 0.0
                        
                        if max_month_high > entry_price:
                            is_profitable = "Yes"
                        else:
                            is_profitable = "No"
                        
                        # Daily fallback if Monthly says No
                        if is_profitable == "No":
                            # Check daily data for more precision
                            if check_profit_with_daily(ticker, entry_price, entry_date.strftime('%Y-%m-%d')):
                                is_profitable = "Yes (Verified by Daily)"
                        
                        all_signals.append({
                            'Ticker': ticker,
                            'Signal_Date': signal_date.strftime('%Y-%m-%d'),
                            'Entry_Date': entry_date.strftime('%Y-%m-%d'),
                            'Entry_Price': entry_price,
                            'SMA200_at_Signal': float(sma200_val) if pd.notnull(sma200_val) else None,
                            'Close_at_Signal': current_close,
                            'Passed_MA200': passed_ma200,
                            'Max_High_Observed': max_month_high,
                            'Is_Profitable': is_profitable,
                            'Method': 'Monthly+Weekly Fallback+MA200'
                        })
            except Exception as e:
                # Log or skip if specific row processing fails
                continue
        
    if all_signals:
        df_out = pd.DataFrame(all_signals)
        out_path = 'n225_ma200_screened_report.csv'
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\nFinal Report saved to {out_path}")
        print(f"Total Tickers Analyzed: {len(tickers)}")
        print(f"Total Signals Recorded: {len(df_out)}")
    else:
        print("No signals found for any ticker.")

if __name__ == "__main__":
    main()
         