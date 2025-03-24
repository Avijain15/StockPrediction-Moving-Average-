# Stock Prediction Platform using Moving Averages
_A Python-based stock prediction tool using various moving average strategies._

## Features

### 📊 Data Fetching
- Retrieves **historical stock data** for the past 100 days.
- Fetches **real-time stock prices** with minute-level granularity.
- Obtains **Nifty index (NSEI) data** for market sentiment analysis.

### 🔍 Moving Average Strategies
#### 1️⃣ Simple Moving Average (SMA)
> Average of closing prices over a user-specified period.

#### 2️⃣ Linearly Weighted Moving Average (LWMA)
> Weights recent prices more heavily.

#### 3️⃣ Exponentially Smoothed Moving Average (EMA)
> Applies exponential smoothing to prioritize recent data.

#### 4️⃣ Two Averages Crossover
> Compares **5-day and 20-day SMAs** for trend identification.

#### 5️⃣ Triple Crossover
> Analyzes **5-day, 10-day, and 20-day SMAs** for strong trend signals.

#### 6️⃣ 4-9-18-20 Day Moving Average
> Uses four SMAs to detect buy/sell opportunities.

#### 7️⃣ 10-day and 50-day SMA Comparison
> Identifies **bullish or bearish trends** based on crossover.

### 📈 Market Sentiment Analysis
- Determines market sentiment (**Bullish, Bearish, Sideways**) using the **Nifty index’s 50-day SMA** trend over the past 5 days.

### 🔮 Crossover Prediction
- Analyzes **historical crossovers** of **10-day or 50-day SMAs** to predict **average percentage price changes** over the next 5 days.
- Segmented by **market sentiment**.

### 🛠️ User Interaction
- **Interactive command-line interface** for selecting stock symbols and moving average types.
- Displays **current prices, moving average values, formulas, and recommendations**.

### 📊 Visualization
- Plots **historical closing prices and moving averages** for the last **30 days** using **matplotlib**.

---

### ✅ Example Output
```bash
Stock: RELIANCE.NS
Current Price: ₹2,300
5-day SMA: ₹2,250
20-day SMA: ₹2,200
Recommendation: 📈 Buy (Uptrend Detected)
