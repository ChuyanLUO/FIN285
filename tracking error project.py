#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:08:30 2018

@author: jingxiaowang
"""
import pandas as pd  
import numpy as np
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() 
import datetime 
import warnings
warnings.filterwarnings("ignore")
import scipy
from scipy import stats
from pandas import ExcelWriter
from pandas import ExcelFile
from scipy import optimize

# weights
TickerNWeights = pd.read_excel('weight.xlsx', sheet_name='Sheet1', header=0)
wts_index = TickerNWeights['Weight']
# download price
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))
start_dt = datetime.datetime(2008, 10, 31)
end_dt = datetime.datetime(2018, 10, 31)
# tickers = ['ACAD','ALKS','ALNY','ALXN','AMGN','ARRY','BIIB','BMRN','CELG','EXEL','FOLD','GHDX','GILD','HALO','ILMN','IMMU','INCY','IONS','JAZZ','LGND','MDCO','MYGN','MYL','NBIX','NKTR','OPK','REGN','RGEN','SGEN','SHPG','SRPT','TECH','UTHR','VRTX']
tickers = TickerNWeights['Ticker']
stock_data = getDataBatch(tickers, start_dt, end_dt)
daily_close_px = stock_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
# Calculate returns
daily_return = daily_close_px.pct_change().dropna()
num_periods, num_stock = daily_return.shape
# write excel
daily_close_px.to_csv('TEdata.csv', header=True, index=True)

#B. Find out weights of index

#C. Forecast covariance matrix
#1. #MA


##2. EWMA


#D. Build ETF
#1. Pick Assets

#2. Build covariance matrix

#3. Partitioning

#4. Cross validation

#5. Adjust ETF and repeat 
