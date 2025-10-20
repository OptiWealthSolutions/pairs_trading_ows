import yfinance as yf
import datetime as dt
import pandas as pd
import os 
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def getData(tickers):
    data = yf.download(tickers,period="20y", interval="1d")['Close']
    data["return_1"] = data[tickers[0]].pct_change()
    data["return_2"] = data[tickers[1]].pct_change()
    data = data.dropna()
    return data

tickers = ['PEP','KO']
data  = getData(tickers)
print(data)

def makereturnregression(data):
    window = 60
    betas, residuals = [], []

    for i in range(window, len(data)):
        y = data['return_2'].iloc[i-window:i]
        X = sm.add_constant(data['return_1'].iloc[i-window:i])
        model = sm.OLS(y, X).fit()
        beta = model.params['return_1']
        resid = model.resid.iloc[-1]  # dernier résidu
        betas.append(beta)
        residuals.append(resid)

    data = data.iloc[window:].copy()
    data['beta'] = betas
    data['residual'] = residuals

    data['Xk'] = data['residual'].cumsum() #on obtient une suite des residus prêt pour la prochaine regression
    return data

# On estime :
# X_{t+1} = a + b X_t + ζ_t

# Puis, à partir de a, b, on en déduit les paramètres du processus O-U :
# θ = -\ln(b) * 252,\quad μ = a / (1 - b),\quad σ = \sqrt{\frac{Var(ζ) * 2θ}{1 - b^2}}

def OU_process(Xk):
    X_t = data['Xk'].shift(1).dropna()
    X_t1 = data['Xk'][1:]
    model_ou = sm.OLS(X_t1, sm.add_constant(X_t)).fit()

    a = model_ou.params['const']
    b = model_ou.params['Xk']
    zeta = model_ou.resid

    theta = -np.log(b) * 252
    mu = a / (1 - b)
    sigma = np.sqrt(np.var(zeta) * 2 * theta / (1 - b**2))
    sigma_eq = np.sqrt(np.var(zeta) * 2 / (2 * theta))

    return theta,mu,sigma,sigma_eq

data= makereturnregression(data)
Xk = data["Xk"]
theta,mu,sigma,sigma_eq = OU_process(Xk)

data['z_score'] = (data['Xk']-mu)/sigma_eq
# At this stage, we can use the standardized version
# of ( ) X t , called Z-score as trading signal. This factor
# measures how far ( ) X t deviates from its mean level and is
# a valid measure across all securities since it is
# dimensionless. More details of signal will be given later.
data['signal'] = 0
s = -mu * np.sqrt(2 * theta) / sigma
lower_band = 0.5*s
upper_band = 1.2*s
entry = upper_band
exit = lower_band

data.loc[data['z_score'] > entry, 'signal'] = -1  # short spread
data.loc[data['z_score'] < -entry, 'signal'] = 1  # long spread
data.loc[data['z_score'].abs() < exit, 'signal'] = 0  # exit position

# Calcul des poids proportionnels à beta
data['w1'] = 1 / (data['beta'] + 1)
data['w2'] = data['beta'] / (data['beta'] + 1)

data['strategy_return'] = data['signal'].shift(1) * (data['w1'] * data['return_1'] - data['w2'] * data['return_2'])
data['cum_pnl'] = (1 + data['strategy_return']).cumprod() - 1

plt.plot(data['cum_pnl'])
plt.show()


plt.plot(data['w1'])
plt.plot(data['w2'])
plt.show()


