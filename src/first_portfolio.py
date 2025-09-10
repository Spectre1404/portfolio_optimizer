import yfinance as yf 
import pandas as pd 
import matplotlib.pyplot as plt 

stock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'BLK']
portfolio_weights = [0.30, 0.20, 0.20, 0.30]

portfolio_data= {}

for ticker in stock_tickers:
    stock = yf.Ticker(ticker)
    portfolio_data[ticker] = stock.history(period = "3mo")


closing_prices = pd.DataFrame()
for ticker in stock_tickers:
    closing_prices[ticker] = portfolio_data[ticker]['Close']

print("Portfolio closing prices (last 10 days):")
print(closing_prices.tail(10))


# Calculating Returns

daily_returns = closing_prices.pct_change().dropna()
print("\nDaily returns (first 10 rows):")
print(daily_returns.head(10))
portfolio_returns = (daily_returns * portfolio_weights).sum(axis=1)

print("Portfolio Stats:")
print(f"Average daily return: {portfolio_returns.mean():.4f}")
print(f"Daily volatility: {portfolio_returns.std():.4f}")

results = pd.DataFrame({
    'daily_returns': portfolio_returns,
    'cumulative_returns': (1 + portfolio_returns).cumprod()
})

print(results)
results.to_csv('portfolio_returns.csv')
print("results csv saved")

