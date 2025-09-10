import yfinance as yf
import pandas as pd 


msft = yf.Ticker("MSFT")
msft_info= msft.info
print(f"Company: {msft_info['longName']}")
print(f"Current Price: ${msft_info['regularMarketPrice']}")

msft_history = msft.history(period="1mo")
print("Recent price data:")
print(msft_history.tail())

print("Data columns:", msft_history.columns.tolist())
print("Data shape:", msft_history.shape)

highest_price = msft_history['High'].max()
lowest_price = msft_history['Low'].min()
average_price = msft_history['Close'].mean()

print(f"Highest price in last month: ${highest_price:.2f}")
print(f"Lowest price in last month: ${lowest_price:.2f}")
print(f"Average closing price: ${average_price:.2f}")

msft_history.to_csv('msft_stock_data.csv')
print("Data saved")