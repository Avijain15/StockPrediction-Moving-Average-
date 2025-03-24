import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to fetch historical data up to yesterday
def fetch_historical_data(symbol, period="100d"):
    """Fetch historical stock data for a given symbol."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            raise ValueError("No historical data available for this symbol.")
        return hist
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

# Function to fetch the current real-time price
def get_current_price(symbol):
    """Fetch the latest real-time price for a stock."""
    try:
        stock = yf.Ticker(symbol)
        latest = stock.history(period="1d", interval="1m").iloc[-1]
        return latest['Close']
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return None

# Simple Moving Average (SMA)
def calculate_sma(data, period):
    """Calculate the Simple Moving Average for a given period."""
    return data['Close'].rolling(window=period).mean()

# Linearly Weighted Moving Average (LWMA)
def calculate_lwma(data, period):
    """Calculate the Linearly Weighted Moving Average for a given period."""
    weights = np.arange(1, period + 1)  # Linear weights: 1, 2, ..., period
    lwma = data['Close'].rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return lwma

# Exponentially Smoothed Moving Average (EMA)
def calculate_ema(data, period):
    """Calculate the Exponential Moving Average for a given period."""
    return data['Close'].ewm(span=period, adjust=False).mean()

# Function to calculate market sentiment based on Nifty's SMA trend
def calculate_sentiment(sma, k=5, threshold=0.005):
    """Determine market sentiment (Bullish, Bearish, Sideways) based on SMA trend."""
    pct_change = (sma - sma.shift(k)) / sma.shift(k)
    sentiment = pd.Series(index=sma.index, dtype='object')
    sentiment[pct_change > threshold] = 'Bullish'
    sentiment[pct_change < -threshold] = 'Bearish'
    sentiment[(pct_change >= -threshold) & (pct_change <= threshold)] = 'Sideways'
    return sentiment

# Function to analyze historical crossovers and calculate average percentage changes
def analyze_historical_crossovers(hist, ma, sentiment, n=5):
    """Analyze historical crossovers and compute average percentage changes by sentiment."""
    crossover_above = (hist['Close'].shift(1) < ma.shift(1)) & (hist['Close'] > ma)
    crossover_below = (hist['Close'].shift(1) > ma.shift(1)) & (hist['Close'] < ma)
    future_pct_change = (hist['Close'].shift(-n) / hist['Close'] - 1) * 100
    crossovers = pd.DataFrame({
        'Date': hist.index,
        'Crossover_Above': crossover_above,
        'Crossover_Below': crossover_below,
        'Sentiment': sentiment.reindex(hist.index),
        'Future_Pct_Change': future_pct_change
    })
    crossovers = crossovers.dropna(subset=['Sentiment', 'Future_Pct_Change'])
    avg_changes = {}
    for sent in ['Bullish', 'Bearish', 'Sideways']:
        above_changes = crossovers[(crossovers['Crossover_Above']) & (crossovers['Sentiment'] == sent)]['Future_Pct_Change']
        below_changes = crossovers[(crossovers['Crossover_Below']) & (crossovers['Sentiment'] == sent)]['Future_Pct_Change']
        if not above_changes.empty:
            avg_changes[(sent, 'Above')] = above_changes.mean()
        if not below_changes.empty:
            avg_changes[(sent, 'Below')] = below_changes.mean()
    return avg_changes

# Plotting function for historical data and moving averages
def plot_ma(hist, ma, title, ma_label="MA"):
    """Plot the closing price and moving averages."""
    plt.figure(figsize=(12, 6))
    plt.plot(hist['Close'][-30:], label='Closing Price')  # Last 30 days for clarity
    if isinstance(ma, dict):
        for key, value in ma.items():
            plt.plot(value[-30:], label=key)
    else:
        plt.plot(ma[-30:], label=ma_label)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main program
def main():
    print("Welcome to the Stock Prediction Platform (Indian Stocks Only)")
    symbol = input("Enter the stock symbol (e.g., RELIANCE.NS for NSE): ").upper()
    if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
        print("Please use .NS for NSE stocks or .BO for BSE stocks (e.g., RELIANCE.NS).")
        return

    # Fetch stock historical data
    hist = fetch_historical_data(symbol)
    if hist is None:
        return

    # Fetch Nifty historical data for market sentiment
    nifty_hist = fetch_historical_data("^NSEI")
    if nifty_hist is None:
        print("Error fetching Nifty data. Cannot proceed with sentiment analysis.")
        return
    nifty_sma50 = calculate_sma(nifty_hist, 50)
    nifty_sentiment = calculate_sentiment(nifty_sma50)

    while True:
        print("\nSelect Moving Average Type:")
        print("a: Simple Moving Average (SMA)")
        print("b: Linearly Weighted Moving Average (LWMA)")
        print("c: Exponentially Smoothed Moving Average (EMA)")
        print("d: Two Averages (5-day and 20-day SMA)")
        print("e: Triple Crossover Moving Average")
        print("f: 4-9-18-20 Day Moving Average")
        print("g: 10-day and 50-day SMA Comparison")
        print("h: Predict movement after MA crossover")
        print("q: Quit")
        choice = input("Enter your choice: ").lower()

        if choice == 'q':
            print("Exiting the platform. Goodbye!")
            break

        current_price = get_current_price(symbol)
        if current_price is None:
            continue
        print(f"\nCurrent Price of {symbol}: ₹{current_price:.2f}")

        if choice in ['a', 'b', 'c']:
            period = int(input(f"Enter the period (in days) for {choice.upper()}: "))
            if period > len(hist):
                print(f"Error: Period ({period}) exceeds available data ({len(hist)} days). Try a smaller period.")
                continue

            if choice == 'a':  # SMA
                ma = calculate_sma(hist, period)
                formula = f"SMA({period}) = (Sum of last {period} closing prices) / {period}"
                last_ma = ma.iloc[-1]
                recommendation = "Buy" if current_price > last_ma else "Sell"
                plot_title = f"{symbol} - SMA ({period}-day)"
                plot_ma(hist, ma, plot_title, f"SMA ({period}-day)")

            elif choice == 'b':  # LWMA
                ma = calculate_lwma(hist, period)
                formula = f"LWMA({period}) = (P1*1 + P2*2 + ... + P{period}*{period}) / (1 + 2 + ... + {period})"
                last_ma = ma.iloc[-1]
                recommendation = "Buy" if current_price > last_ma else "Sell"
                plot_title = f"{symbol} - LWMA ({period}-day)"
                plot_ma(hist, ma, plot_title, f"LWMA ({period}-day)")

            elif choice == 'c':  # EMA
                ma = calculate_ema(hist, period)
                smoothing = 2 / (period + 1)
                formula = f"EMA({period}) = (Close * {smoothing:.3f}) + (Previous EMA * {1 - smoothing:.3f})"
                last_ma = ma.iloc[-1]
                recommendation = "Buy" if current_price > last_ma else "Sell"
                plot_title = f"{symbol} - EMA ({period}-day)"
                plot_ma(hist, ma, plot_title, f"EMA ({period}-day)")

            print(f"\nMathematical Formula: {formula}")
            print(f"Recommendation: {recommendation}")

        elif choice == 'd':  # Two Averages (5-day and 20-day SMA)
            ma5 = calculate_sma(hist, 5)
            ma20 = calculate_sma(hist, 20)
            formula = "Two Averages: SMA(5) and SMA(20) compared for crossovers"
            last_ma5, last_ma20 = ma5.iloc[-1], ma20.iloc[-1]
            recommendation = "Buy" if last_ma5 > last_ma20 else "Sell" if last_ma5 < last_ma20 else "Hold"
            plot_title = f"{symbol} - 5-day vs 20-day SMA"
            plot_ma(hist, {"SMA(5)": ma5, "SMA(20)": ma20}, plot_title)
            print(f"\nMathematical Formula: {formula}")
            print(f"Recommendation: {recommendation}")

        elif choice == 'e':  # Triple Crossover (5, 10, 20 days)
            ma5 = calculate_sma(hist, 5)
            ma10 = calculate_sma(hist, 10)
            ma20 = calculate_sma(hist, 20)
            formula = "Triple Crossover: SMA(5), SMA(10), SMA(20) order determines trend"
            last_ma5, last_ma10, last_ma20 = ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1]
            if last_ma5 > last_ma10 > last_ma20:
                recommendation = "Buy"
            elif last_ma5 < last_ma10 < last_ma20:
                recommendation = "Sell"
            else:
                recommendation = "Hold"
            plot_title = f"{symbol} - Triple Crossover (5-10-20)"
            plot_ma(hist, {"SMA(5)": ma5, "SMA(10)": ma10, "SMA(20)": ma20}, plot_title)
            print(f"\nMathematical Formula: {formula}")
            print(f"Recommendation: {recommendation}")

        elif choice == 'f':  # 4-9-18-20 Day Moving Average
            ma4 = calculate_sma(hist, 4)
            ma9 = calculate_sma(hist, 9)
            ma18 = calculate_sma(hist, 18)
            ma20 = calculate_sma(hist, 20)
            formula = "4-9-18-20 MA: SMA(4), SMA(9), SMA(18), SMA(20) order analysis"
            last_ma4, last_ma9, last_ma18, last_ma20 = ma4.iloc[-1], ma9.iloc[-1], ma18.iloc[-1], ma20.iloc[-1]
            if last_ma4 > last_ma9 > last_ma18 > last_ma20:
                recommendation = "Buy"
            elif last_ma4 < last_ma9 < last_ma18 < last_ma20:
                recommendation = "Sell"
            else:
                recommendation = "Hold"
            plot_title = f"{symbol} - 4-9-18-20 Day MA"
            plot_ma(hist, {"SMA(4)": ma4, "SMA(9)": ma9, "SMA(18)": ma18, "SMA(20)": ma20}, plot_title)
            print(f"\nMathematical Formula: {formula}")
            print(f"Recommendation: {recommendation}")

        elif choice == 'g':  # 10-day and 50-day SMA Comparison
            ma10 = calculate_sma(hist, 10)
            ma50 = calculate_sma(hist, 50)
            formula = "10-day and 50-day SMA Comparison: If 10-day SMA > 50-day SMA, bullish; else bearish"
            last_ma10, last_ma50 = ma10.iloc[-1], ma50.iloc[-1]
            recommendation = "Buy" if last_ma10 > last_ma50 else "Sell"
            plot_title = f"{symbol} - 10-day vs 50-day SMA"
            plot_ma(hist, {"SMA(10)": ma10, "SMA(50)": ma50}, plot_title)
            print(f"\nMathematical Formula: {formula}")
            print(f"Recommendation: {recommendation}")

        elif choice == 'h':  # Predict movement after MA crossover
            print("Select MA for crossover prediction:")
            print("1: 10-day SMA")
            print("2: 50-day SMA")
            ma_choice = input("Enter 1 or 2: ")
            if ma_choice == '1':
                period = 10
            elif ma_choice == '2':
                period = 50
            else:
                print("Invalid choice.")
                continue
            ma = calculate_sma(hist, period)
            avg_changes = analyze_historical_crossovers(hist, ma, nifty_sentiment)
            yesterday_close = hist['Close'].iloc[-1]
            yesterday_ma = ma.iloc[-1]
            current_price = get_current_price(symbol)
            if current_price is None:
                continue
            current_sentiment = nifty_sentiment.iloc[-1]
            if pd.isna(current_sentiment):
                print("Cannot determine current market sentiment due to insufficient data.")
                continue
            if yesterday_close < yesterday_ma and current_price > yesterday_ma:
                crossover_type = 'Above'
            elif yesterday_close > yesterday_ma and current_price < yesterday_ma:
                crossover_type = 'Below'
            else:
                print("No crossover detected today.")
                continue
            key = (current_sentiment, crossover_type)
            if key in avg_changes:
                avg_pct_change = avg_changes[key]
                print(f"Based on historical data, after a crossover {crossover_type.lower()} the {period}-day SMA in a {current_sentiment.lower()} market, the stock has moved an average of {avg_pct_change:.2f}% over the next 5 days.")
            else:
                print(f"No historical data available for crossover {crossover_type.lower()} in a {current_sentiment.lower()} market.")

        else:
            print("Invalid choice. Please select a valid option.")
            continue

if __name__ == "__main__":
    main()