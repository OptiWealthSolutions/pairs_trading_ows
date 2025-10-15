import pandas as pd
import os 
from utils.data_engineer import DataEngineer
import numpy as np
import statsmodels as sm
from statsmodels import coint, OLS

#pipe line road map


class Strategy():
    def __init__(self):
        pass

    def computeSignal(self):
        pass

    def getcoint(data):
        score, p_value, _ = coint(data[0], data[1])
        return p_value

    def createPairs(self):
        pass
    
    # Linear Regression for Estimating Weights and Parameters
    def computeResiduals(X,y):
        model = sm.OLS(X,y) 
        results = model.fit()

    # make a new regression to estimate theta, sigma and mu in the O-U process
    def method_moment(X,dt):
        deltaX = np.diff(X)
        X = X[:-1]
        mu = X.mean()
        exog = (mu - X) * dt
        model = OLS(endog=deltaX, exog=exog)
        res = model.fit()
        theta = res.params[0]
        resid = deltaX - theta * exog
        sigma = resid.std() / np.sqrt(dt)
        return theta,mu,sigma

#weight :  (1/(beta+1), (beta/beta+1))
#discret version of Xk = ∑ residuals
# make a new regression to estimate theta, sigma and mu in the O-U process