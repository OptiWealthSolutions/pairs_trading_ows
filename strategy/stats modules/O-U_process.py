import pandas as pd
import numpy as np
from statsmodels.api import OLS
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from main import Strategy

#estimation of Xt with the sum of the residuals from the OLS regression in main.py
st = Strategy()

def llh(params,X,dt):
    theta, mu ,sigma = params
    deltaX = np.diff(X)
    X = X[:-1]
    f = -0.5 * np.log(2 * np.pi * sigma**2 * dt) - ((deltaX - theta*(mu - X)*dt)**2) / (2 * sigma**2 * dt)
    res = -f.sum() #scipy ne peut que faire des minimisation donc on passe par l'inverse de la fonction '-f'
    if np.isnan(res) or np.isinf(res):
        res = 1
    return res

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
