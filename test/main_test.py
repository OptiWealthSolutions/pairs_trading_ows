import yfinance as yf
import datetime as dt
import pandas as pd
import os 
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller

def getData(tickers):
    data = yf.download(tickers,period="20y", interval="1d")['Close']
    data["return_1"] = data[0].pct_change()
    data["return_2"] = data[1].pct_change()
    return data

tickers = ['AAPL','MSFT']
data  = getData(tickers)

def makereturnregression(data):
    X = data['return_1']
    y = data['return_2']

    model = sm.OLS(X,y)

    results = model.fit()
    beta = results.beta
    resid = results.resid
    w1 = (beta/(beta+1))
    w2 = (1/(beta+1))

    weights = [w1,w2]
    return resid, weights

regression = makereturnregression(data)
print(regression)

