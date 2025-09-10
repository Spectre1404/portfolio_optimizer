## Portfolio Optimizer - Week 1

###  Summary
- Downloaded and analyzed real historical stock data using yfinance
- Created a portfolio with 4 stocks: AAPL, GOOGL, MSFT, BLK
- Calculated daily returns, cumulative returns, and portfolio stats
- Saved outputs to `portfolio_returns.csv` and `portfolio_prices.png`

### Files Created
- `src/first_portfolio.py`: Main analysis script
- `portfolio_returns.csv`: Daily and cumulative returns
- `portfolio_prices.png`: Line plot of stock prices
- `msft_stock_data.csv`: Test run for single stock
- `README.md`: Project notes

### Key Learnings
- Portfolio = group of stocks with weights
- Daily return = % change in price
- Volatility = standard deviation of daily returns
- Cumulative return shows how value compounds over time

### Challenges
- Syntax issues with script filenames
- Learned to inspect dictionaries like `ticker.info`
- Understood why `pct_change()` is essential