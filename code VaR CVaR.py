import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import norm, t
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------------
# CONFIGURATION
# -----------------------------------

stockList = ['RELIANCE', 'TCS', 'ICICIBANK', 'INFY']
stockCounts = [15,10,8,5]  # Adjust as per your holdings
initialInvestment = 10000
alpha = 5
Time = 100  # Time horizon in days

# -----------------------------------
# LOAD DATA
# -----------------------------------

# Load historical data
excel_path = r"C:\Users\Admin\Desktop\Resume\FINAL\indian_stocks_data Final 4 stocks.xlsx"
sheets = pd.read_excel(excel_path, sheet_name=stockList)

# Extract and combine closing prices
adj_close_data = pd.concat({ticker: df.set_index('Date')['Close'] for ticker, df in sheets.items()}, axis=1)
adj_close_data.index = pd.to_datetime(adj_close_data.index)
stockData = adj_close_data.dropna()

# -----------------------------------
# PORTFOLIO CONSTRUCTION
# -----------------------------------

returns = stockData.pct_change().dropna()
latestPrices = stockData.iloc[-1]
stockCounts = stockCounts[:len(latestPrices)]

stockValues = np.array(stockCounts) * latestPrices
weights = stockValues / stockValues.sum()

meanReturns = returns.mean()
covMatrix = returns.cov()
portfolioReturns = returns.dot(weights)

orig_return = np.sum(meanReturns * weights)
orig_vol = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))
orig_sharpe = orig_return / orig_vol

# -----------------------------------
# SHARPE RATIO OPTIMIZATION
# -----------------------------------

def neg_sharpe(weights, meanReturns, covMatrix):
    port_return = np.sum(meanReturns * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))
    return -port_return / port_vol

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(stockList)))
opt_result = minimize(neg_sharpe, weights, args=(meanReturns, covMatrix), method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_result.x

opt_return = np.sum(meanReturns * optimal_weights)
opt_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(covMatrix, optimal_weights)))
opt_sharpe = opt_return / opt_vol

# -----------------------------------
# PORTFOLIO PERFORMANCE (Original)
# -----------------------------------

def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    pReturn = np.sum(meanReturns * weights) * Time
    pStd = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(Time)
    return pReturn, pStd

pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)

# -----------------------------------
# HISTORICAL VaR & CVaR
# -----------------------------------

def historicalVaR(returns, alpha=5):
    return np.percentile(returns, alpha)

def historicalCVaR(returns, alpha=5):
    var_level = historicalVaR(returns, alpha)
    return returns[returns <= var_level].mean()

hVaR = historicalVaR(portfolioReturns, alpha) * np.sqrt(Time)
hCVaR = historicalCVaR(portfolioReturns, alpha) * np.sqrt(Time)

# -----------------------------------
# PARAMETRIC VaR & CVaR (Normal & t)
# -----------------------------------

def var_parametric(mu, sigma, dist='normal', alpha=5, dof=6):
    if dist == 'normal':
        return -(mu + norm.ppf(1 - alpha / 100) * sigma)
    elif dist == 't-distribution':
        return -(mu + np.sqrt((dof - 2) / dof) * t.ppf(1 - alpha / 100, dof) * sigma)

def cvar_parametric(mu, sigma, dist='normal', alpha=5, dof=6):
    if dist == 'normal':
        z = norm.ppf(1 - alpha / 100)
        return -(mu + sigma * norm.pdf(z) / (alpha / 100))
    elif dist == 't-distribution':
        z = t.ppf(1 - alpha / 100, dof)
        return -(mu + sigma * t.pdf(z, dof) * (dof - 2 + z**2) / (dof * (alpha / 100)))

normVaR = var_parametric(pRet, pStd, 'normal', alpha)
normCVaR = cvar_parametric(pRet, pStd, 'normal', alpha)
tVaR = var_parametric(pRet, pStd, 't-distribution', alpha)
tCVaR = cvar_parametric(pRet, pStd, 't-distribution', alpha)

# -----------------------------------
# MONTE CARLO SIMULATION
# -----------------------------------

simulations = 10000
meanM = np.full((Time, len(weights)), meanReturns.values)
portfolio_sims = np.zeros((Time, simulations))
L = np.linalg.cholesky(covMatrix)

for i in range(simulations):
    Z = np.random.normal(size=(Time, len(weights)))
    dailyReturns = meanM + Z @ L.T
    portfolioPath = np.cumprod(1 + dailyReturns @ weights)
    portfolio_sims[:, i] = portfolioPath * initialInvestment

final_values = pd.Series(portfolio_sims[-1, :])
mcVaR = initialInvestment - np.percentile(final_values, 100 - alpha)
mcCVaR = initialInvestment - final_values[final_values <= np.percentile(final_values, 100 - alpha)].mean()

# -----------------------------------
# BACKTESTING HISTORICAL VAR
# -----------------------------------

backtest_breaches = (portfolioReturns < -historicalVaR(portfolioReturns, alpha)).sum()
total_days = portfolioReturns.shape[0]
breach_rate = backtest_breaches / total_days

# -----------------------------------
# DIVERSIFICATION EFFECT
# -----------------------------------

individual_var = -returns.quantile(alpha / 100) * latestPrices * stockCounts
sum_individual_var = individual_var.sum()

# -----------------------------------
# RESULTS SUMMARY
# -----------------------------------

summary = pd.DataFrame({
    'Method': ['Historical', 'Parametric (Normal)', 'Parametric (t)', 'Monte Carlo'],
    'VaR (INR)': [hVaR * initialInvestment, normVaR * initialInvestment, tVaR * initialInvestment, mcVaR],
    'CVaR (INR)': [hCVaR * initialInvestment, normCVaR * initialInvestment, tCVaR * initialInvestment, mcCVaR]
})

print("\n----- Portfolio Risk Summary (95% CI) -----")
print(summary.to_string(index=False))
#print(f"\nBacktesting VaR breaches: {backtest_breaches} out of {total_days} days ({breach_rate:.2%})")
#print(f"\nSum of Individual Stock VaRs: ₹{sum_individual_var:.2f}")
#print(f"Portfolio VaR (Historical): ₹{hVaR * initialInvestment:.2f}")

print("\n----- Original Portfolio Weights -----")
for i, stock in enumerate(stockList):
    print(f"{stock}: {weights[i]:.2%}")
print(f"Expected Annualized Return: {orig_return * 252:.2%}")
print(f"Expected Annualized Volatility: {orig_vol * np.sqrt(252):.2%}")
print(f"Sharpe Ratio: {orig_sharpe * np.sqrt(252):.2f}")

print("\n----- Optimal Portfolio Weights (Sharpe Ratio Maximization) -----")
for i, stock in enumerate(stockList):
    print(f"{stock}: {optimal_weights[i]:.2%}")
print(f"Expected Annualized Return: {opt_return * 252:.2%}")
print(f"Expected Annualized Volatility: {opt_vol * np.sqrt(252):.2%}")
print(f"Sharpe Ratio: {opt_sharpe * np.sqrt(252):.2f}")

# -----------------------------------
# PLOT MONTE CARLO SIMULATIONS
# -----------------------------------

plt.figure(figsize=(10, 4))
plt.plot(portfolio_sims, alpha=0.1, color='blue')
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.xlabel('Days')
plt.ylabel('Portfolio Value (INR)')
plt.grid(True)
plt.tight_layout()
plt.savefig("monte_carlo_simulation.png", dpi=300)
plt.show()


#====================================================================
# COLOURFUL MONTE CARLO SIMULATIONS
#====================================================================
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'portfolio_sims' is a 2D NumPy array: shape = (num_days, num_simulations)
num_simulations = portfolio_sims.shape[1]

plt.figure(figsize=(10, 4))

# Plot each simulation with a unique color from a colormap
colors = plt.cm.magma(np.linspace(0, 1, num_simulations))  # You can use other colormaps too

for i in range(num_simulations):
    plt.plot(portfolio_sims[:, i], color=colors[i], alpha=0.8, linewidth=1)

plt.title('Monte Carlo Simulation of Portfolio Value')
plt.xlabel('Days')
plt.ylabel('Portfolio Value (INR)')
plt.grid(True)
plt.tight_layout()
plt.savefig("monte_carlo_simulation_colorful.png", dpi=300)
plt.show()




