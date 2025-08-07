
# 📉 Portfolio Risk Analysis using VaR and CVaR

## 📘 Project Title
**Comprehensive Portfolio Risk Management using Historical, Parametric, and Monte Carlo VaR & CVaR Methods**

## 📊 Overview
This project analyzes the risk of a portfolio composed of Indian stocks using multiple Value at Risk (VaR) and Conditional Value at Risk (CVaR) techniques. It also includes portfolio optimization based on the Sharpe ratio and evaluates performance through simulation and backtesting.

## 🎯 Objectives
- Compute Historical, Parametric (Normal and t-distribution), and Monte Carlo-based VaR and CVaR.
- Optimize portfolio allocation by maximizing Sharpe ratio.
- Backtest Historical VaR and calculate breach rate.
- Simulate portfolio value using Monte Carlo for probabilistic insights.

## 📁 Files Included
- `full code VaR CVaR.py` — Main Python script for analysis and visualization.
- `indian_stocks_data Final 4 stocks.xlsx` — Historical stock data for 4 Indian companies.
- `monte_carlo_simulation.png` & `monte_carlo_simulation_colorful.png` — Output plots from Monte Carlo simulations.

## 🧪 Methodologies Used
- **Historical VaR & CVaR:** Based on empirical return distributions.
- **Parametric VaR & CVaR:** Using normal and t-distributions.
- **Monte Carlo Simulation:** Simulated future price paths for portfolio evaluation.
- **Sharpe Ratio Optimization:** Portfolio allocation through SLSQP optimization.
- **Backtesting:** Validation of VaR predictions through historical breaches.

## 📈 Portfolio Details
- Stocks used: RELIANCE, TCS, ICICIBANK, INFY
- Input data: Daily close prices
- Portfolio construction based on quantity held per stock
- Time horizon: 100 trading days

## 📦 Requirements
- Python ≥ 3.7
- Packages: `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl`

```bash
pip install pandas numpy scipy matplotlib openpyxl
```

## ▶️ How to Run
1. Place the Excel file in the working directory.
2. Update the file path in the script if needed.
3. Run `full code VaR CVaR.py` in your Python environment.
4. View the summary output and simulation plots.

## 📊 Outputs
- Risk summaries for VaR/CVaR across methods
- Optimal portfolio weights vs original
- Backtest results with breach rate
- Monte Carlo portfolio path visualizations

## 📌 Notes
- Historical prices must include 'Date' and 'Close' columns per sheet.
- Data frequency: Daily
- Risk confidence level: 95% (alpha = 5)

<img width="1272" height="466" alt="image" src="https://github.com/user-attachments/assets/8615a178-a429-4d27-82ad-b904229016df" />
