import pandas as pd
import os 
from utils.data_engineer import DataEngineer
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import yfinance as yf
#pipe line road map


class Strategy():
    def __init__(self):
        self.dt = 1/252

    def computeSignal(self):
        pass
    
    def getdata(self, tickers):
        data = yf.download(tickers, period='20y',interval="1d")['Close']
        return data
    
    def getcoint(self, df):
        score, p_value, _ = coint(df.iloc[:,0], df.iloc[:,1])
        print(p_value)
        if p_value >= 0.05:
                print( "pas cointegrées")
        else:
                print("cointegrées")
        return p_value

    def computeregression(self, data):
        X, y = data.iloc[:,0], data.iloc[:,1]
        model = sm.OLS(y, sm.add_constant(X))
        results = model.fit()
        residuals = results.resid
        data['resid'] = residuals
        return data
    
    def ZScore(self, df):
        df['mean'] = df['resid'].mean()
        df['std'] = df['resid'].std()
        df['z_score'] = (df['resid'] - df['mean']) / df['std']
        return df

    def computethreshold(self, df):
        #threshold 
        upper_band = df.iloc[:,0].rolling(window=60).std() * 1.2
        lower_band = -df.iloc[:,0].rolling(window=60).std() * 1.2
        return upper_band, lower_band

    # make a new regression to estimate theta, sigma and mu in the O-U process
    def method_moment(self, X):
        deltaX = np.diff(X)
        X = X[:-1]
        mu = X.mean()
        exog = (mu - X) * self.dt
        model = sm.OLS(deltaX, exog)
        res = model.fit()
        theta = res.params[0]
        resid = deltaX - theta * exog
        sigma = resid.std() / np.sqrt(self.dt)
        return theta, mu, sigma


#weight :  (1/(beta+1), (beta/beta+1))
#discret version of Xk = ∑ residuals
# make a new regression to estimate theta, sigma and mu in the O-U process

if __name__ == "__main__":
    strat = Strategy()
    tickers = ['AAPL', 'MSFT']
    data = strat.getdata(tickers)
    p_value = strat.getcoint(data)
    if p_value < 0.05:
        data = strat.computeregression(data)
        data = strat.ZScore(data)
        upper_band, lower_band = strat.computethreshold(data)
        print("Upper band head:\n", upper_band.head())
        print("Lower band head:\n", lower_band.head())
        theta, mu, sigma = strat.method_moment(data['resid'].dropna().values)
        print(f"Theta: {theta}, Mu: {mu}, Sigma: {sigma}")
    else:
        print("The series are not cointegrated, skipping regression and further analysis.")