import yfinance as yf
import pandas as pd
import datetime as dt

class DataEngineer():
    def __init__(self, *args, **kwargs):
        pass

    def getDataLoad(ticker,period,interval):
        df = yf.download(ticker, period=period, progress=False,interval=interval)['Close']
        return df
    
    def getDataClean(df):
        df = df.copy()
        df = df.dropna()
        return df
    