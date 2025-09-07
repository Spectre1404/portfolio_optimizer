# Variables and data types
company_name = "Apple Inc."
stock_price = 175.50
shares_owned = 100

# Lists (our future portfolio)
portfolio_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
portfolio_weights = [0.25, 0.30, 0.25, 0.20]

# Simple calculations
portfolio_value = stock_price * shares_owned
print(f"Portfolio value: ${portfolio_value}")

# Loop practice
for stock in portfolio_stocks:
    print(f"Stock: {stock}")